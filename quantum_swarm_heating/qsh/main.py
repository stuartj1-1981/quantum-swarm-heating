from logging import config
from .config import HOUSE_CONFIG
from .simulation import sim_step
from .rl_model import ActorCritic, train_rl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from datetime import datetime, timedelta, timezone
import time
import logging
import os
import json
import requests
from collections import defaultdict, deque
import signal
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")

def main():
    global prev_flow, prev_mode, prev_demand, action_counter

    graph = build_dfan_graph(HOUSE_CONFIG)
    state_dim = 15
    action_dim = 2
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    try:
        train_rl(model, optimizer, episodes=200)
    except Exception as e:
        logging.error(f"RL training failed, continuing with untrained model: {e}")

    while True:
        try:
            optimal_mode, optimal_flow, total_demand_adjusted, prev_flow, prev_mode, prev_demand = sim_step(
                graph,
                None,
                HOUSE_CONFIG,
                model,
                optimizer,
                action_counter,
                prev_flow,
                prev_mode,
                prev_demand
            )
            action_counter += 1

            if optimal_mode not in ['off', 'heat']:
                logging.warning(f"Invalid optimal_mode '{optimal_mode}', forcing 'heat'")
                optimal_mode = 'heat'
        except Exception as e:
            logging.error(f"Top-level control loop error: {e}. Entering failsafe mode.")
            try:
                hp_flow_cfg = HOUSE_CONFIG['hp_flow_service']
                flow_data = dict(hp_flow_cfg.get('base_data', {}))
                flow_data.update({
                    "device_id": hp_flow_cfg.get("device_id"),
                    "flow_temp": 40.0
                })
                set_ha_service(hp_flow_cfg['domain'], hp_flow_cfg['service'], flow_data)

                hvac_cfg = HOUSE_CONFIG['hp_hvac_service']
                hvac_data = {
                    "device_id": hvac_cfg.get("device_id"),
                    "hvac_mode": "heat"
                }
                set_ha_service(hvac_cfg['domain'], hvac_cfg['service'], hvac_data)
            except Exception as inner_e:
                logging.error(f"Failsafe application failed: {inner_e}")

        time.sleep(30)

if __name__ == "__main__":
    main()
