/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Top-level threads for gNodeB
 */

#define _GNU_SOURCE
#undef MALLOC //there are two conflicting definitions, so we better make sure we don't use it at all

#include <inttypes.h>
#include <stdlib.h>
#include <fcntl.h> // for SEEK_SET
#include <pthread.h> // for pthread_join
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include "common/utils/LOG/log.h"
#include "common/utils/system.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "openair1/PHY/NR_TRANSPORT/nr_dlsch.h"
#include "openair1/PHY/NR_TRANSPORT/nr_ulsch.h"
#include "NR_PHY_INTERFACE/NR_IF_Module.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_RU.h"
#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_common.h"
#include "PHY/impl_defs_nr.h"
#include "SCHED_NR/phy_frame_config_nr.h"
#include "SCHED_NR/sched_nr.h"
#include "assertions.h"
#include "common/ran_context.h"
#include "common/utils/LOG/log.h"
#include "executables/softmodem-common.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include "nfapi_nr_interface_scf.h"
#include "notified_fifo.h"
#include "thread-pool.h"
#include "time_meas.h"
#include "utils.h"


#define TICK_TO_US(ts) (ts.trials==0?0:ts.diff/ts.trials)
#define L1STATSSTRLEN 16384
#define GNB_TIMELINE_MAX 500000

extern _Thread_local uint64_t gnb_sched_lock_acquired_ns_tls;

extern _Thread_local int gnb_ul_collect_count_tls;
extern _Thread_local int gnb_ul_first_target_slot_tls;
extern _Thread_local int gnb_ul_first_k2_tls;

typedef struct {
  uint64_t sched_start_ns;
  uint64_t sched_lock_acquired_ns;
  uint64_t sched_end_ns;

  int ul_collect_count;
  int ul_first_target_slot;
  int ul_first_k2;

  int frame_tx;
  int slot_tx;

  int frame_rx;
  int slot_rx;
} gnb_tx_timeline_entry_t;

typedef struct {
  uint64_t rx_func_start_ns;

  uint64_t phy_start_ns;
  uint64_t phy_end_ns;              /* native PHY completion */
  uint64_t phy_release_target_ns;   /* requested absolute release time */
  uint64_t phy_release_ns;          /* actual release time */

  uint64_t ul_ind_start_ns;
  uint64_t ul_ind_end_ns;

  uint64_t l1_complete_ns;

  int frame_rx;
  int slot_rx;
} gnb_rx_timeline_entry_t;

static gnb_tx_timeline_entry_t gnb_tx_timeline[GNB_TIMELINE_MAX];
static gnb_rx_timeline_entry_t gnb_rx_timeline[GNB_TIMELINE_MAX];

static uint64_t gnb_start_mono_ns = 0;
static uint64_t gnb_start_realtime_ns = 0;

static uint64_t gnb_end_mono_ns = 0;
static uint64_t gnb_end_realtime_ns = 0;
static uint64_t gnb_phy_release_target_ns = 0;

static _Atomic uint32_t gnb_tx_timeline_count = 0;
static _Atomic uint32_t gnb_rx_timeline_count = 0;
static _Atomic int gnb_timeline_atexit_registered = 0;

static inline uint64_t gnb_timeline_now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

  return ((uint64_t)ts.tv_sec * 1000000000ULL)
       + (uint64_t)ts.tv_nsec;
}

static void gnb_init_phy_release_target(void)
{
  //helper function to read the environment variable GNB_PHY_RELEASE_TARGET_US and set gnb_phy_release_target_ns accordingly
  const char *env = getenv("GNB_PHY_RELEASE_TARGET_US");

  if (env == NULL || env[0] == '\0') {
    gnb_phy_release_target_ns = 0;
    fprintf(stderr,
            "gNB PHY release delay: disabled "
            "(GNB_PHY_RELEASE_TARGET_US not set)\n");
    return;
  }

  errno = 0;
  char *end = NULL;
  unsigned long long target_us = strtoull(env, &end, 10);

  AssertFatal(errno == 0 &&
              end != env &&
              *end == '\0',
              "Invalid GNB_PHY_RELEASE_TARGET_US='%s'\n",
              env);

  AssertFatal(target_us <= UINT64_MAX / 1000ULL,
              "GNB_PHY_RELEASE_TARGET_US too large\n");

  gnb_phy_release_target_ns = (uint64_t)target_us * 1000ULL;

  fprintf(stderr,
          "gNB PHY release target: %llu us\n",
          target_us);
}

static void gnb_wait_until_raw_ns(uint64_t target_ns)
{
  while (1) {
    uint64_t now_ns = gnb_timeline_now_ns();

    if (now_ns >= target_ns)
      return;

    uint64_t remaining_ns = target_ns - now_ns;

    struct timespec req = {
      .tv_sec = remaining_ns / 1000000000ULL,
      .tv_nsec = remaining_ns % 1000000000ULL
    };

    while (nanosleep(&req, &req) == -1 && errno == EINTR) {
      /* Continue sleeping for the remaining interval. */
    }
  }
}

static inline uint64_t gnb_realtime_now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  return ((uint64_t)ts.tv_sec * 1000000000ULL)
       + (uint64_t)ts.tv_nsec;
}

static void gnb_timeline_dump(void)
{
  gnb_end_mono_ns = gnb_timeline_now_ns();
  gnb_end_realtime_ns = gnb_realtime_now_ns();
  const char *prefix = getenv("GNB_TIMELINE_PREFIX");
  if (prefix == NULL || prefix[0] == '\0')
    prefix = "/tmp/gnb_timeline";

  char tx_path[512];
  char rx_path[512];
  char clock_path[512];

  snprintf(tx_path, sizeof(tx_path), "%s_tx.csv", prefix);
  snprintf(rx_path, sizeof(rx_path), "%s_rx.csv", prefix);
  snprintf(clock_path, sizeof(clock_path), "%s_clock.csv", prefix);

  /* ---------------- TX / scheduler timeline ---------------- */

  FILE *txf = fopen(tx_path, "w");
  if (txf != NULL) {
    fprintf(txf,
        "sched_start_ns,sched_lock_acquired_ns,sched_end_ns,"
        "ul_collect_count,ul_first_target_slot,ul_first_k2,"
        "frame_tx,slot_tx,frame_rx,slot_rx\n");

    uint32_t n_tx = atomic_load(&gnb_tx_timeline_count);
    if (n_tx > GNB_TIMELINE_MAX)
      n_tx = GNB_TIMELINE_MAX;

    for (uint32_t i = 0; i < n_tx; i++) {
      const gnb_tx_timeline_entry_t *e = &gnb_tx_timeline[i];

      fprintf(txf,
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
        "%d,%d,%d,"
        "%d,%d,%d,%d\n",
        e->sched_start_ns,
        e->sched_lock_acquired_ns,
        e->sched_end_ns,
        e->ul_collect_count,
        e->ul_first_target_slot,
        e->ul_first_k2,
        e->frame_tx,
        e->slot_tx,
        e->frame_rx,
        e->slot_rx);
      }

    fclose(txf);
    fprintf(stderr, "gNB timeline: wrote %u TX entries to %s\n",
            n_tx, tx_path);
  } else {
    fprintf(stderr, "gNB timeline: could not open %s\n", tx_path);
  }

  /* ---------------- RX / UL timeline ---------------- */

  FILE *rxf = fopen(rx_path, "w");
  if (rxf != NULL) {
    fprintf(rxf,
        "rx_func_start_ns,"
        "phy_start_ns,phy_end_ns,"
        "phy_release_target_ns,phy_release_ns,"
        "ul_ind_start_ns,ul_ind_end_ns,"
        "l1_complete_ns,"
        "frame_rx,slot_rx\n");

    uint32_t n_rx = atomic_load(&gnb_rx_timeline_count);
    if (n_rx > GNB_TIMELINE_MAX)
      n_rx = GNB_TIMELINE_MAX;

    for (uint32_t i = 0; i < n_rx; i++) {
      const gnb_rx_timeline_entry_t *e = &gnb_rx_timeline[i];

      fprintf(rxf,
        "%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ",%" PRIu64 ","
        "%" PRIu64 ","
        "%d,%d\n",
        e->rx_func_start_ns,
        e->phy_start_ns,
        e->phy_end_ns,
        e->phy_release_target_ns,
        e->phy_release_ns,
        e->ul_ind_start_ns,
        e->ul_ind_end_ns,
        e->l1_complete_ns,
        e->frame_rx,
        e->slot_rx);
    }

    fclose(rxf);
    fprintf(stderr, "gNB timeline: wrote %u RX entries to %s\n",
            n_rx, rx_path);
  } else {
    fprintf(stderr, "gNB timeline: could not open %s\n", rx_path);
  }

  uint32_t observed_tx = atomic_load(&gnb_tx_timeline_count);
  uint32_t observed_rx = atomic_load(&gnb_rx_timeline_count);

  if (observed_tx > GNB_TIMELINE_MAX)
    fprintf(stderr,
            "gNB timeline: dropped %u TX entries\n",
            observed_tx - GNB_TIMELINE_MAX);

  if (observed_rx > GNB_TIMELINE_MAX)
    fprintf(stderr,
            "gNB timeline: dropped %u RX entries\n",
            observed_rx - GNB_TIMELINE_MAX);

  FILE *clockf = fopen(clock_path, "w");

  if (clockf != NULL) {
    fprintf(clockf,
            "start_mono_ns,start_realtime_ns,"
            "end_mono_ns,end_realtime_ns\n");

    fprintf(clockf,
            "%" PRIu64 ",%" PRIu64 ","
            "%" PRIu64 ",%" PRIu64 "\n",
            gnb_start_mono_ns,
            gnb_start_realtime_ns,
            gnb_end_mono_ns,
            gnb_end_realtime_ns);

    fclose(clockf);

    fprintf(stderr,
            "gNB timeline: wrote clock mapping to %s\n",
            clock_path);
  }
}

static void rx_func(processingData_L1_t *param);

static void tx_func(processingData_L1tx_t *info)
{
  int frame_tx = info->frame;
  int slot_tx = info->slot;
  int frame_rx = info->frame_rx;
  int slot_rx = info->slot_rx;

  uint32_t tx_trace_idx = atomic_fetch_add(&gnb_tx_timeline_count, 1);
  gnb_tx_timeline_entry_t *tx_trace = NULL;

  if (tx_trace_idx < GNB_TIMELINE_MAX) {
    tx_trace = &gnb_tx_timeline[tx_trace_idx];

    tx_trace->frame_tx = frame_tx;
    tx_trace->slot_tx = slot_tx;
    tx_trace->frame_rx = frame_rx;
    tx_trace->slot_rx = slot_rx;
  }

  LOG_D(NR_PHY, "%d.%d running tx_func\n", frame_tx, slot_tx);
  PHY_VARS_gNB *gNB = info->gNB;
  NR_IF_Module_t *ifi = gNB->if_inst;
  nfapi_nr_config_request_scf_t *cfg = &gNB->gNB_config;

  T(T_GNB_PHY_DL_TICK, T_INT(gNB->Mod_id), T_INT(frame_tx), T_INT(slot_tx));

  if (slot_rx == 0) {
    reset_active_stats(gNB, frame_rx);
    reset_active_ulsch(gNB, frame_rx);
  }

  clear_slot_beamid(gNB, slot_tx);

  nfapi_nr_slot_indication_scf_t ind = {.sfn = frame_tx, .slot = slot_tx};
  start_meas(&gNB->slot_indication_stats);

  static NR_Sched_Rsp_t sched_response;

  uint64_t sched_start_ns = gnb_timeline_now_ns();

  /* Reset before entering the scheduler.
  * gNB_dlsch_ulsch_scheduler() will set this immediately after
  * acquiring sched_lock. */
  gnb_sched_lock_acquired_ns_tls = 0;

  gnb_ul_collect_count_tls = 0;
  gnb_ul_first_target_slot_tls = -1;
  gnb_ul_first_k2_tls = -1;

  ifi->NR_slot_indication(&ind, &sched_response);

  uint64_t sched_end_ns = gnb_timeline_now_ns();

  if (tx_trace != NULL) {
    tx_trace->sched_start_ns = sched_start_ns;
    tx_trace->sched_lock_acquired_ns = gnb_sched_lock_acquired_ns_tls;
    tx_trace->sched_end_ns = sched_end_ns;

    tx_trace->ul_collect_count = gnb_ul_collect_count_tls;
    tx_trace->ul_first_target_slot = gnb_ul_first_target_slot_tls;
    tx_trace->ul_first_k2 = gnb_ul_first_k2_tls;
  }

  stop_meas(&gNB->slot_indication_stats);

  info->gNB = gNB;

  // At this point, MAC scheduler just ran, including scheduling
  // PRACH/PUCCH/PUSCH, so trigger RX chain processing
  nr_save_ul_tti_req(gNB, &sched_response.UL_tti_req);
  LOG_D(NR_PHY, "Trigger RX for %d.%d\n", frame_rx, slot_rx);
  notifiedFIFO_elt_t *res = newNotifiedFIFO_elt(sizeof(processingData_L1_t), 0, &gNB->resp_L1, NULL);
  processingData_L1_t *syncMsg = NotifiedFifoData(res);
  syncMsg->gNB = gNB;
  syncMsg->frame_rx = frame_rx;
  syncMsg->slot_rx = slot_rx;
  syncMsg->timestamp_tx = info->timestamp_tx;
  res->key = slot_rx;
  pushNotifiedFIFO(&gNB->resp_L1, res);

  int tx_slot_type = nr_slot_select(cfg, frame_tx, slot_tx);
  // TODO check for analog_bf_vendor_ext set to 1 is a workaround while no beam API for beam selection is implemented
  if (tx_slot_type == NR_DOWNLINK_SLOT || tx_slot_type == NR_MIXED_SLOT || get_softmodem_params()->continuous_tx
      || IS_SOFTMODEM_RFSIM || cfg->analog_beamforming_ve.analog_bf_vendor_ext.value) {
    start_meas(&info->gNB->phy_proc_tx);
    phy_procedures_gNB_TX(info->gNB, &sched_response.DL_req, &sched_response.TX_req, &sched_response.UL_dci_req, frame_tx,slot_tx);

    PHY_VARS_gNB *gNB = info->gNB;
    processingData_RU_t syncMsgRU;
    syncMsgRU.frame_tx = frame_tx;
    syncMsgRU.slot_tx = slot_tx;
    syncMsgRU.ru = gNB->RU_list[0];
    syncMsgRU.timestamp_tx = info->timestamp_tx;
    LOG_D(PHY, "gNB: %d.%d : calling RU TX function\n", syncMsgRU.frame_tx, syncMsgRU.slot_tx);
    ru_tx_func((void *)&syncMsgRU);
    stop_meas(&info->gNB->phy_proc_tx);
  }
}

void *L1_rx_thread(void *arg) 
{
  PHY_VARS_gNB *gNB = (PHY_VARS_gNB*)arg;

  while (oai_exit == 0) {
     notifiedFIFO_elt_t *res = pullNotifiedFIFO(&gNB->resp_L1);
     if (res == NULL)
       break;
     processingData_L1_t *info = (processingData_L1_t *)NotifiedFifoData(res);
     start_meas(&gNB->l1_rx_proc);
     rx_func(info);
     stop_meas(&gNB->l1_rx_proc);
     delNotifiedFIFO_elt(res);
  }
  return NULL;
}
// Added for URLLC, requires MAC scheduling to be split from UL indication
void *L1_tx_thread(void *arg) {
  PHY_VARS_gNB *gNB = (PHY_VARS_gNB*)arg;

  while (oai_exit == 0) {
     notifiedFIFO_elt_t *res = pullNotifiedFIFO(&gNB->L1_tx_out);
     if (res == NULL) // stopping condition, happens only when queue is freed
       break;
     processingData_L1tx_t *info = (processingData_L1tx_t *)NotifiedFifoData(res);
     start_meas(&gNB->l1_tx_proc);
     tx_func(info);
     stop_meas(&gNB->l1_tx_proc);
     delNotifiedFIFO_elt(res);
  }
  return NULL;
}

static void rx_func(processingData_L1_t *info)
{
  PHY_VARS_gNB *gNB = info->gNB;
  int frame_rx = info->frame_rx;
  int slot_rx = info->slot_rx;
  uint32_t rx_trace_idx = atomic_fetch_add(&gnb_rx_timeline_count, 1);
  gnb_rx_timeline_entry_t *rx_trace = NULL;

  if (rx_trace_idx < GNB_TIMELINE_MAX) {
    rx_trace = &gnb_rx_timeline[rx_trace_idx];

    rx_trace->frame_rx = frame_rx;
    rx_trace->slot_rx = slot_rx;
    rx_trace->rx_func_start_ns = gnb_timeline_now_ns();
  }
  nfapi_nr_config_request_scf_t *cfg = &gNB->gNB_config;

  T(T_GNB_PHY_UL_TICK, T_INT(gNB->Mod_id), T_INT(frame_rx), T_INT(slot_rx));

  // RX processing
  int rx_slot_type = nr_slot_select(cfg, frame_rx, slot_rx);
  if (rx_slot_type == NR_UPLINK_SLOT || rx_slot_type == NR_MIXED_SLOT) {
    LOG_D(NR_PHY, "%d.%d Starting RX processing\n", frame_rx, slot_rx);

    // UE-specific RX processing for subframe n
    NR_UL_IND_t UL_INFO = {.frame = frame_rx, .slot = slot_rx, .module_id = gNB->Mod_id, .CC_id = gNB->CC_id};
    // Do PRACH RU processing
    UL_INFO.rach_ind.pdu_list = UL_INFO.prach_pdu_indication_list;
    UL_INFO.rach_ind.number_of_pdus = 0;
    // even if processing is late, we might collect all PRACH
    // the last PRACH's frame/slot is when all UE's appear to have accessed
    prach_item_t p;
    while (spsc_q_get(&gNB->prach_l1rx_queue, &p, sizeof(p)))
      L1_nr_prach_procedures(gNB, &p, &UL_INFO.rach_ind);

    //WA: comment rotation in tx/rx
    if (gNB->phase_comp) {
      //apply the rx signal rotation here
      int soffset = (slot_rx % RU_RX_SLOT_DEPTH) * gNB->frame_parms.symbols_per_slot * gNB->frame_parms.ofdm_symbol_size;
      const NR_DL_FRAME_PARMS *fp = &gNB->frame_parms;
      for (int aa = 0; aa < fp->nb_antennas_rx; aa++) {
        const uint max_symb = fp->Ncp == NR_EXTENDED ? 12 : 14;
        for (int sym = 0; sym < max_symb; sym++)
          apply_nr_rotation_symbol_RX(fp->symbols_per_slot,
                                      fp->slots_per_subframe,
                                      fp->timeshift_symbol_rotation,
                                      fp->first_carrier_offset,
                                      gNB->common_vars.rxdataF[aa] + soffset + sym * fp->ofdm_symbol_size,
                                      fp->symbol_rotation[1],
                                      fp->N_RB_UL,
                                      slot_rx,
                                      sym);
      }
    }
    uint64_t phy_start_ns = gnb_timeline_now_ns();

    phy_procedures_gNB_uespec_RX(gNB, frame_rx, slot_rx, &UL_INFO);

    /* Native PHY completion: no artificial delay included. */
    uint64_t phy_end_ns = gnb_timeline_now_ns();

    /* Controlled PHY-result release.
    *
    * GNB_PHY_RELEASE_TARGET_US defines a minimum PHY latency measured
    * from phy_start_ns. For example, with a 1500 us target:
    *
    *   native PHY = 900 us  -> wait about 600 us
    *   native PHY = 1700 us -> no wait
    *
    * Therefore:
    *
    *   effective latency = max(native latency, requested target)
    */
    uint64_t phy_release_target_ns = 0;

    if (gnb_phy_release_target_ns > 0) {
      phy_release_target_ns = phy_start_ns + gnb_phy_release_target_ns;

      if (phy_end_ns < phy_release_target_ns)
        gnb_wait_until_raw_ns(phy_release_target_ns);
    }

    /* Actual instant at which the PHY result is allowed to proceed to MAC. */
    uint64_t phy_release_ns = gnb_timeline_now_ns();

    if (rx_trace != NULL) {
      rx_trace->phy_start_ns = phy_start_ns;
      rx_trace->phy_end_ns = phy_end_ns;
      rx_trace->phy_release_target_ns = phy_release_target_ns;
      rx_trace->phy_release_ns = phy_release_ns;
    }

    // Call the scheduler
    start_meas(&gNB->ul_indication_stats);

    uint64_t ul_ind_start_ns = gnb_timeline_now_ns();

    gNB->if_inst->NR_UL_indication(&UL_INFO);

    uint64_t ul_ind_end_ns = gnb_timeline_now_ns();

    if (rx_trace != NULL) {
      rx_trace->ul_ind_start_ns = ul_ind_start_ns;
      rx_trace->ul_ind_end_ns = ul_ind_end_ns;
    }

    stop_meas(&gNB->ul_indication_stats);

    notifiedFIFO_elt_t *res = newNotifiedFIFO_elt(sizeof(processingData_L1_t), 0, &gNB->L1_rx_out, NULL);
    processingData_L1_t *syncMsg = NotifiedFifoData(res);
    syncMsg->gNB = gNB;
    syncMsg->frame_rx = frame_rx;
    syncMsg->slot_rx = slot_rx;
    res->key = slot_rx;
    LOG_D(NR_PHY, "Signaling completion for %d.%d (mod_slot %d) on L1_rx_out\n", frame_rx, slot_rx, slot_rx % RU_RX_SLOT_DEPTH);
    if (rx_trace != NULL) {
      rx_trace->l1_complete_ns = gnb_timeline_now_ns();
    }
    pushNotifiedFIFO(&gNB->L1_rx_out, res);
  }

}

static size_t dump_L1_meas_stats(PHY_VARS_gNB *gNB, RU_t *ru, char *output, size_t outputlen) {
  const char *begin = output;
  const char *end = output + outputlen;
  output += print_meas_log(&gNB->l1_tx_proc, "L1 Tx job", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->l1_rx_proc, "L1 Rx job", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->phy_proc_tx, "L1 Tx processing", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->dlsch_encoding_stats, "DLSCH encoding", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->dlsch_scrambling_stats, "DLSCH scrambling", NULL, NULL, output, end-output);
  output += print_meas_log(&gNB->dlsch_modulation_stats, "DLSCH modulation", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->dlsch_pdsch_generation_stats, "PDSCH generation", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->phy_proc_rx, "L1 Rx processing", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->ts_deinterleave, "UL segment deinterleaving", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->ts_rate_unmatch, "UL segment rate recovery", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->ts_ldpc_decode, "UL segments decoding", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->ul_indication_stats, "UL Indication", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->slot_indication_stats, "Slot Indication", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->rx_pusch_stats, "PUSCH inner-receiver", NULL, NULL, output, end - output);
  output += print_meas_log(&gNB->rx_prach, "PRACH RX", NULL, NULL, output, end - output);
  if (ru->feprx)
    output += print_meas_log(&ru->ofdm_demod_stats, "feprx", NULL, NULL, output, end - output);

  bool full_slot = ru->half_slot_parallelization == 0;
  if (ru->feptx_prec) {
    output += print_meas_log(&ru->precoding_stats,
                             full_slot ? "feptx_prec (per port)" : "feptx_prec (per port, half_slot)",
                             NULL,
                             NULL,
                             output,
                             end - output);
  }

  if (ru->feptx_ofdm) {
    output += print_meas_log(&ru->txdataF_copy_stats,"txdataF_copy",NULL,NULL, output, end - output);
    output += print_meas_log(&ru->ofdm_mod_stats,
                             full_slot ? "feptx_ofdm (per port)" : "feptx_ofdm (per port, half_slot)",
                             NULL,
                             NULL,
                             output,
                             end - output);
    output += print_meas_log(&ru->ofdm_total_stats,"feptx_total",NULL,NULL, output, end - output);
    output += print_meas_log(&ru->txdataF_copy_stats, "txdataF_copy", NULL, NULL, output, end - output);
  }

  output += print_meas_log(&ru->tx_fhaul,"tx_fhaul",NULL,NULL, output, end - output);

  return output - begin;
}

void *nrL1_stats_thread(void *param) {
  PHY_VARS_gNB     *gNB      = (PHY_VARS_gNB *)param;
  RU_t *ru = RC.ru[0];
  char output[L1STATSSTRLEN];
  memset(output,0,L1STATSSTRLEN);
  wait_sync("L1_stats_thread");
  FILE *fd=fopen("nrL1_stats.log","w");
  if (!fd) {
    LOG_W(NR_PHY, "Cannot open nrL1_stats.log: %d, %s\n", errno, strerror(errno));
    return NULL;
  }

  reset_meas(&gNB->l1_tx_proc);
  reset_meas(&gNB->l1_rx_proc);
  reset_meas(&gNB->phy_proc_tx);
  reset_meas(&gNB->dlsch_encoding_stats);
  reset_meas(&gNB->phy_proc_rx);
  reset_meas(&gNB->ts_deinterleave);
  reset_meas(&gNB->ts_rate_unmatch);
  reset_meas(&gNB->ts_ldpc_decode);
  reset_meas(&gNB->ul_indication_stats);
  reset_meas(&gNB->slot_indication_stats);
  reset_meas(&gNB->rx_pusch_stats);
  reset_meas(&gNB->dlsch_scrambling_stats);
  reset_meas(&gNB->dlsch_modulation_stats);
  reset_meas(&gNB->dlsch_pdsch_generation_stats);
  while (!oai_exit) {
    sleep(1);
    if (ftruncate(fileno(fd), 0) != 0 || fseek(fd, 0, SEEK_SET) != 0) {
      LOG_E(NR_MAC, "error while writing nrL1_stats.log: %d, %s\n", errno, strerror(errno));
      break;
    }
    dump_nr_I0_stats(fd,gNB);
    dump_pdsch_stats(fd,gNB);
    dump_pusch_stats(fd,gNB);
    dump_L1_meas_stats(gNB, ru, output, L1STATSSTRLEN);
    fprintf(fd,"%s\n",output);
    fflush(fd);
  }
  fclose(fd);
  return(NULL);
}

void init_gNB_Tpool(int inst)
{
  gnb_init_phy_release_target();
  AssertFatal(NFAPI_MODE == NFAPI_MODE_PNF || NFAPI_MODE == NFAPI_MONOLITHIC,
              "illegal NFAPI_MODE %d (%s): it cannot have an L1\n",
              NFAPI_MODE,
              nfapi_get_strmode());

  PHY_VARS_gNB *gNB;
  gNB = RC.gNB[inst];
  gNB_L1_proc_t *proc = &gNB->proc;
  
  // ULSCH decoding threadpool
  initTpool(get_softmodem_params()->threadPoolConfig, &gNB->threadPool, cpumeas(CPUMEAS_GETSTATE));

  // L1 RX result FIFO
  initNotifiedFIFO(&gNB->resp_L1);
  // L1 TX result FIFO
  initNotifiedFIFO(&gNB->L1_tx_out);
  initNotifiedFIFO(&gNB->L1_rx_out);
  if (atomic_exchange(&gnb_timeline_atexit_registered, 1) == 0) {
    gnb_start_mono_ns = gnb_timeline_now_ns();
    gnb_start_realtime_ns = gnb_realtime_now_ns();

    atexit(gnb_timeline_dump);
  }

  // create the RX thread responsible for RX processing start event (resp_L1 msg queue), then launch rx_func()
  threadCreate(&gNB->L1_rx_thread, L1_rx_thread, (void *)gNB, "L1_rx_thread", gNB->L1_rx_thread_core, OAI_PRIORITY_RT_MAX);
  // create the TX thread responsible for TX processing start event (L1_tx_out msg queue), then launch tx_func()
  threadCreate(&gNB->L1_tx_thread, L1_tx_thread, (void *)gNB, "L1_tx_thread", gNB->L1_tx_thread_core, OAI_PRIORITY_RT_MAX);

  if (!IS_SOFTMODEM_NOSTATS)
    threadCreate(&proc->L1_stats_thread, nrL1_stats_thread, (void *)gNB, "L1_stats", -1, OAI_PRIORITY_RT_LOW);
}

void term_gNB_Tpool(int inst) {
  PHY_VARS_gNB *gNB = RC.gNB[inst];
  abortNotifiedFIFO(&gNB->resp_L1);
  pthread_join(gNB->L1_rx_thread, NULL);
  abortNotifiedFIFO(&gNB->L1_tx_out);
  pthread_join(gNB->L1_tx_thread, NULL);

  abortTpool(&gNB->threadPool);
  abortNotifiedFIFO(&gNB->L1_rx_out);

  gNB_L1_proc_t *proc = &gNB->proc;
  pthread_join(proc->L1_stats_thread, NULL);
}

/// eNB kept in function name for nffapi calls, TO FIX
void init_eNB_afterRU(void)
{
  for (int inst = 0; inst < RC.nb_nr_L1_inst; inst++) {
    PHY_VARS_gNB *gNB = RC.gNB[inst];
    phy_init_nr_gNB(gNB);

    // map antennas and PRACH signals to gNB RX
    if (0) AssertFatal(gNB->num_RU>0,"Number of RU attached to gNB %d is zero\n",gNB->Mod_id);

    LOG_D(NR_PHY, "Mapping RX ports from %d RUs to gNB %d\n", gNB->num_RU, gNB->Mod_id);
    int aa = 0;
    for (int ru_id = 0; ru_id < gNB->num_RU; ru_id++) {
      AssertFatal(gNB->RU_list[ru_id]->common.rxdataF != NULL, "RU %d : common.rxdataF is NULL\n", gNB->RU_list[ru_id]->idx);
      for (int i = 0; i < gNB->RU_list[ru_id]->nb_rx; aa++, i++) {
        LOG_I(PHY, "Attaching RU %d antenna %d to gNB antenna %d\n", gNB->RU_list[ru_id]->idx, i, aa);
        gNB->common_vars.rxdataF[aa] = (c16_t *)gNB->RU_list[ru_id]->common.rxdataF[i];
      }
    }
    /* TODO: review this code, there is something wrong.
     * In monolithic mode, we come here with nb_antennas_rx == 0
     * (not tested in other modes).
     */
    //init_precoding_weights(RC.gNB[inst]);
    init_gNB_Tpool(inst);
  }
}

/**
 * @brief Initialize gNB struct in RAN context
 */
void init_gNB()
{
  LOG_I(NR_PHY, "Initializing gNB RAN context: RC.nb_nr_L1_inst = %d \n", RC.nb_nr_L1_inst);
  if (RC.gNB == NULL) {
    RC.gNB = (PHY_VARS_gNB **)calloc_or_fail(RC.nb_nr_L1_inst, sizeof(PHY_VARS_gNB *));
    LOG_D(NR_PHY, "gNB L1 structure RC.gNB allocated @ %p\n", RC.gNB);
  }

  for (int inst = 0; inst < RC.nb_nr_L1_inst; inst++) {
    // Allocate L1 instance
    if (RC.gNB[inst] == NULL) {
      RC.gNB[inst] = (PHY_VARS_gNB *)calloc_or_fail(1, sizeof(PHY_VARS_gNB));
      LOG_D(NR_PHY, "[nr-gnb.c] gNB structure RC.gNB[%d] allocated @ %p\n", inst, RC.gNB[inst]);
    }
    PHY_VARS_gNB *gNB = RC.gNB[inst];
    LOG_D(NR_PHY, "Initializing gNB %d\n", inst);

    // Init module ID
    gNB->Mod_id = inst;

    // Register MAC interface module
    AssertFatal((gNB->if_inst = NR_IF_Module_init(inst)) != NULL, "Cannot register interface");

    LOG_I(NR_PHY, "Registered with MAC interface module (%p)\n", gNB->if_inst);
    gNB->if_inst->NR_PHY_config_req = nr_phy_config_request;

    gNB->prach_energy_counter = 0;
    gNB->chest_time = get_softmodem_params()->chest_time;
    gNB->chest_freq = get_softmodem_params()->chest_freq;
  }
}

void stop_gNB(int nb_inst) {
  for (int inst=0; inst<nb_inst; inst++) {
    LOG_I(PHY,"Killing gNB %d processing threads\n",inst);
    term_gNB_Tpool(inst);
  }
}
