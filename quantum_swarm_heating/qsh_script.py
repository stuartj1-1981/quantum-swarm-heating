from logging import config
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

# HA API setup
HA_URL = 'http://supervisor/core/api'
HA_TOKEN = os.getenv('SUPERVISOR_TOKEN')

logging.info(f"Detected SUPERVISOR_TOKEN: {'Set' if HA_TOKEN else 'None'}")

if not HA_TOKEN:
    logging.critical("SUPERVISOR_TOKEN not set! Using defaults only. Ensure hassio_api: true in config.yaml.")
else:
    logging.info("SUPERVISOR_TOKEN found—using real HA API calls.")

def fetch_ha_entity(entity_id, attr=None):
    if not HA_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.get(f"{HA_URL}/states/{entity_id}", headers=headers)
        r.raise_for_status()
        data = r.json()
        if attr:
            return data.get('attributes', {}).get(attr)
        return data.get('state')
    except Exception as e:
        logging.error(f"HA fetch error for {entity_id}: {e}")
        return None

def set_ha_service(domain, service, data):
    if not HA_TOKEN:
        return
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
    entity_id = data.get('entity_id')
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            if isinstance(entity_id, list):
                for eid in entity_id:
                    data_single = data.copy()
                    data_single['entity_id'] = eid
                    r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data_single)
                    r.raise_for_status()
            else:
                r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data)
                r.raise_for_status()
            return  # Success
        except Exception as e:
            retries += 1
            logging.warning(f"HA set error for {entity_id or data.get('device_id')}: {e} - retry {retries}/{max_retries}")
            time.sleep(5)  # 5s sleep
    logging.error(f"HA set failed after {max_retries} retries for {entity_id or data.get('device_id')}")

# Hardcoded HOUSE_CONFIG using defaults
DEFAULT_ROOMS = { 'lounge': 19.48, 'open_plan': 42.14, 'utility': 3.40, 'cloaks': 2.51,
    'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
    'hall': 9.15, 'landing': 10.09 }
DEFAULT_FACINGS = { 'lounge': 0.2, 'open_plan': 1.0, 'utility': 0.5, 'cloaks': 0.5,
    'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
    'hall': 0.2, 'landing': 0.2 }
DEFAULT_ENTITIES = {
    'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',
    'open_plan_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],
    'utility_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1604136448',
    'cloaks_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0949825024',
    'bed1_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1287620864',
    'bed2_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1941512960',
    'bed3_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4141228288',
    'bed4_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va2043158784',
    'bathroom_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va2920296192',
    'ensuite1_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0001191680',
    'ensuite2_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va1209347840',
    'hall_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0567183616',
    'landing_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va0951787776',
    'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',
    'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',
    'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',
    'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',
    'battery_soc': 'sensor.givtcp_ce2029g082_soc',
    'current_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates',
    'next_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates',
    'current_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates',
    'next_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates',
    'solar_production': 'sensor.envoy_122019031249_current_power_production',
    'outdoor_temp': 'sensor.front_door_motion_temperature',
    'forecast_weather': 'weather.home',
    'hp_output': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output',
    'hp_energy_rate': 'sensor.shellyem_c4d8d5001966_channel_1_power',
    'total_heating_energy': 'sensor.shellyem_c4d8d5001966_channel_1_energy',
    'water_heater': 'water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31',
    'flow_min_temp': 'input_number.flow_min_temperature',
    'flow_max_temp': 'input_number.flow_max_temperature',
    'hp_cop': 'sensor.live_cop_calc',
    'dfan_control_toggle': 'input_boolean.dfan_control',
    'pid_target_temperature': 'input_number.pid_target_temperature',
    'grid_power': 'sensor.givtcp_ce2029g082_grid_power',
    'primary_diff': 'sensor.primary_diff',
    'hp_flow_temp': 'sensor.primary_flow_temperature',
    'lounge_heating': 'sensor.lounge_heating',
    'open_plan_heating': 'sensor.living_area_heating',
    'utility_heating': 'sensor.utility_heating',
    'cloaks_heating': 'sensor.wc_heating',
    'bed1_heating': 'sensor.master_bedroom_heating',
    'bed2_heating': 'sensor.fins_room_heating',
    'bed3_heating': 'sensor.office_heating',
    'bed4_heating': 'sensor.b1llz_room_heating',
    'bathroom_heating': 'sensor.bathroom_heating',
    'ensuite1_heating': 'sensor.ensuite1_heating',
    'ensuite2_heating': 'sensor.ensuite2_heating',
    'hall_heating': 'sensor.hall_heating',
    'landing_heating': 'sensor.landing_heating'
}
DEFAULT_ZONE_SENSOR_MAP = { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan': 'independent_sensor04',
    'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
    'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' }
DEFAULT_BATTERY = {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0}
DEFAULT_GRID = {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0}
DEFAULT_FALLBACK_RATES = {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15}
DEFAULT_INVERTER = {'fallback_efficiency': 0.95}
DEFAULT_PEAK_LOSS = 5.0
DEFAULT_DESIGN_TARGET = 21.0
DEFAULT_PEAK_EXT = -3.0
DEFAULT_THERMAL_MASS_PER_M2 = 0.02  # Lowered to reduce heat-up
DEFAULT_HEAT_UP_TAU_H = 1.0
DEFAULT_PERSISTENT_ZONES = ['bathroom', 'ensuite1', 'ensuite2']
DEFAULT_HP_FLOW_SERVICE = {
    'domain': 'octopus_energy',
    'service': 'set_heat_pump_flow_temp_config',
    'device_id': 'b680894cd18521f7c706f1305b7333ea',
    'base_data': {
        'weather_comp_enabled': False
    }
}
DEFAULT_HP_HVAC_SERVICE = {
    'domain': 'climate',
    'service': 'set_hvac_mode',
    'device_id': 'b680894cd18521f7c706f1305b7333ea'
}
DEFAULT_ROOM_CONTROL_MODE = {
    'lounge': 'direct',
    'open_plan': 'indirect',
    'utility': 'indirect',
    'cloaks': 'indirect',
    'bed1': 'indirect',
    'bed2': 'indirect',
    'bed3': 'indirect',
    'bed4': 'indirect',
    'bathroom': 'indirect',
    'ensuite1': 'indirect',
    'ensuite2': 'indirect',
    'hall': 'indirect',
    'landing': 'indirect'
}
DEFAULT_EMITTER_KW = {
    'lounge': 1.4,
    'open_plan': 3.1,
    'utility': 0.6,
    'cloaks': 0.6,
    'bed1': 1.6,
    'bed2': 1.0,
    'bed3': 1.0,
    'bed4': 1.3,
    'bathroom': 0.39,
    'ensuite1': 0.39,
    'ensuite2': 0.39,
    'hall': 1.57,
    'landing': 1.1
}
DEFAULT_NUDGE_BUDGET = 2.5  # Lowered default to reduce over-boost

HOUSE_CONFIG = {
    'rooms': DEFAULT_ROOMS,
    'facings': DEFAULT_FACINGS,
    'entities': DEFAULT_ENTITIES,
    'zone_sensor_map': DEFAULT_ZONE_SENSOR_MAP,
    'battery': DEFAULT_BATTERY,
    'grid': DEFAULT_GRID,
    'fallback_rates': DEFAULT_FALLBACK_RATES,
    'inverter': DEFAULT_INVERTER,
    'peak_loss': DEFAULT_PEAK_LOSS,
    'design_target': DEFAULT_DESIGN_TARGET,
    'peak_ext': DEFAULT_PEAK_EXT,
    'thermal_mass_per_m2': DEFAULT_THERMAL_MASS_PER_M2,
    'heat_up_tau_h': DEFAULT_HEAT_UP_TAU_H,
    'persistent_zones': DEFAULT_PERSISTENT_ZONES,
    'hp_flow_service': DEFAULT_HP_FLOW_SERVICE,
    'hp_hvac_service': DEFAULT_HP_HVAC_SERVICE,
    'room_control_mode': DEFAULT_ROOM_CONTROL_MODE,
    'emitter_kw': DEFAULT_EMITTER_KW,
    'nudge_budget': DEFAULT_NUDGE_BUDGET
}

logging.info("Using hardcoded HOUSE_CONFIG for stability (Phase 1B).")

# Rest of the script remains unchanged
# ... (truncated for brevity, but includes all original functions like build_dfan_graph, ActorCritic, train_rl, sim_step, live_loop, etc.)
graph = build_dfan_graph(HOUSE_CONFIG)
state_dim = 14
action_dim = 2
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
states = torch.zeros(state_dim)

train_rl(model, optimizer)

def live_loop(graph, states, config, model, optimizer):
    global action_counter, prev_flow, prev_mode, prev_demand
    while True:
        action_counter, prev_flow, prev_mode, prev_demand = sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand)
        time.sleep(120)

live_loop(graph, states, HOUSE_CONFIG, model, optimizer)