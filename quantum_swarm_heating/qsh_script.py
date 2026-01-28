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
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load user options
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")

# HA API setup
HA_URL = 'http://supervisor/core/api'
HA_TOKEN = os.getenv('SUPERVISOR_TOKEN')

logging.info(f"Detected SUPERVISOR_TOKEN: {'Set' if HA_TOKEN else 'None'}")

if not HA_TOKEN:
    logging.critical("SUPERVISOR_TOKEN not set! Using defaults only. Ensure hassio_api: true in config.yaml.")
else:
    logging.info("SUPERVISOR_TOKEN found—using real HA API calls.")

def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        logging.warning(f"Invalid float conversion: {val}")
        return default

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

# Default HOUSE_CONFIG
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

# Configurable reward coefficients
REWARD_COEFFS = {
    'rate_pen': -0.8,
    'cop_bonus': 0.5,
    'heat_up_pen': -0.1,
    'delta_t_bonus': 0.5,
    'cop_low_pen': -0.5,
    'diss_bonus': 0.3,
    'diss_pen': -0.2,
    'demand_stable_bonus': 0.3,
    'flow_stable_bonus': 0.2,
    'volatility_pen_base': 0.2,
    'high_demand_pen': -0.5,
    'high_flow_pen': -1.5,
    'low_open_pen_direct': -2.5,
    'low_open_pen_indirect': -2.0
}

# HOUSE_CONFIG with defaults
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

# Safe override with type checks
def safe_override(key, default):
    if key in user_options:
        val = user_options[key]
        if isinstance(val, type(default)):
            HOUSE_CONFIG[key] = val
            logging.info(f"Overrode {key} from options.json")
        else:
            logging.error(f"Invalid type for {key} in options.json: expected {type(default)}, got {type(val)}. Using default.")

safe_override('rooms', DEFAULT_ROOMS)
safe_override('facings', DEFAULT_FACINGS)
safe_override('entities', DEFAULT_ENTITIES)
safe_override('zone_sensor_map', DEFAULT_ZONE_SENSOR_MAP)
safe_override('battery', DEFAULT_BATTERY)
safe_override('grid', DEFAULT_GRID)
safe_override('fallback_rates', DEFAULT_FALLBACK_RATES)
safe_override('inverter', DEFAULT_INVERTER)
safe_override('peak_loss', DEFAULT_PEAK_LOSS)
safe_override('design_target', DEFAULT_DESIGN_TARGET)
safe_override('peak_ext', DEFAULT_PEAK_EXT)
safe_override('thermal_mass_per_m2', DEFAULT_THERMAL_MASS_PER_M2)
safe_override('heat_up_tau_h', DEFAULT_HEAT_UP_TAU_H)
safe_override('persistent_zones', DEFAULT_PERSISTENT_ZONES)
safe_override('hp_flow_service', DEFAULT_HP_FLOW_SERVICE)
safe_override('hp_hvac_service', DEFAULT_HP_HVAC_SERVICE)
safe_override('room_control_mode', DEFAULT_ROOM_CONTROL_MODE)
safe_override('emitter_kw', DEFAULT_EMITTER_KW)
safe_override('nudge_budget', DEFAULT_NUDGE_BUDGET)

# Validate HOUSE_CONFIG after overrides
def validate_house_config(config):
    errors = []
    if not isinstance(config['rooms'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['rooms'].items()):
        errors.append("Invalid 'rooms': must be dict with str keys and numeric values.")
    if any(v < 0 for v in config['rooms'].values()):
        errors.append("Room areas must be non-negative.")
    if not isinstance(config['facings'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['facings'].items()):
        errors.append("Invalid 'facings': must be dict with str keys and numeric values.")
    if any(v < 0 or v > 1 for v in config['facings'].values()):
        errors.append("Room facings must be between 0 and 1.")
    if not isinstance(config['entities'], dict) or not all(isinstance(k, str) and isinstance(v, (str, list)) for k, v in config['entities'].items()):
        errors.append("Invalid 'entities': must be dict with str keys and str or list values.")
    if not isinstance(config['zone_sensor_map'], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in config['zone_sensor_map'].items()):
        errors.append("Invalid 'zone_sensor_map': must be dict with str keys and str values.")
    if not isinstance(config['battery'], dict) or not all(k in DEFAULT_BATTERY for k in config['battery']):
        errors.append("Invalid 'battery': missing required keys.")
    if not isinstance(config['grid'], dict) or not all(k in DEFAULT_GRID for k in config['grid']):
        errors.append("Invalid 'grid': missing required keys.")
    if not isinstance(config['fallback_rates'], dict) or not all(k in DEFAULT_FALLBACK_RATES for k in config['fallback_rates']):
        errors.append("Invalid 'fallback_rates': missing required keys.")
    if not isinstance(config['inverter'], dict) or not all(k in DEFAULT_INVERTER for k in config['inverter']):
        errors.append("Invalid 'inverter': missing required keys.")
    if not isinstance(config['peak_loss'], (int, float)) or config['peak_loss'] <= 0:
        errors.append("Invalid 'peak_loss': must be positive numeric.")
    if not isinstance(config['design_target'], (int, float)):
        errors.append("Invalid 'design_target': must be numeric.")
    if not isinstance(config['peak_ext'], (int, float)):
        errors.append("Invalid 'peak_ext': must be numeric.")
    if not isinstance(config['thermal_mass_per_m2'], (int, float)) or config['thermal_mass_per_m2'] <= 0:
        errors.append("Invalid 'thermal_mass_per_m2': must be positive numeric.")
    if not isinstance(config['heat_up_tau_h'], (int, float)) or config['heat_up_tau_h'] <= 0:
        errors.append("Invalid 'heat_up_tau_h': must be positive numeric.")
    if not isinstance(config['persistent_zones'], list) or not all(isinstance(z, str) for z in config['persistent_zones']):
        errors.append("Invalid 'persistent_zones': must be list of str.")
    if not isinstance(config['hp_flow_service'], dict) or not all(k in DEFAULT_HP_FLOW_SERVICE for k in config['hp_flow_service']):
        errors.append("Invalid 'hp_flow_service': missing required keys.")
    if not isinstance(config['hp_hvac_service'], dict) or not all(k in DEFAULT_HP_HVAC_SERVICE for k in config['hp_hvac_service']):
        errors.append("Invalid 'hp_hvac_service': missing required keys.")
    if not isinstance(config['room_control_mode'], dict) or not all(isinstance(k, str) and v in ['direct', 'indirect'] for k, v in config['room_control_mode'].items()):
        errors.append("Invalid 'room_control_mode': must be dict with str keys and 'direct' or 'indirect' values.")
    if not isinstance(config['emitter_kw'], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in config['emitter_kw'].items()):
        errors.append("Invalid 'emitter_kw': must be dict with str keys and numeric values.")
    if any(v <= 0 for v in config['emitter_kw'].values()):
        errors.append("Emitter kW must be positive.")
    if not isinstance(config['nudge_budget'], (int, float)) or config['nudge_budget'] < 0:
        errors.append("Invalid 'nudge_budget': must be non-negative numeric.")

    if errors:
        for err in errors:
            logging.error(err)
        raise ValueError("HOUSE_CONFIG validation failed.")
    logging.info("HOUSE_CONFIG validated successfully.")

validate_house_config(HOUSE_CONFIG)

# State encapsulation class
class QSHState:
    def __init__(self):
        self.action_counter = 0
        self.prev_flow = 35.0
        self.prev_mode = 'off'
        self.prev_demand = 0.0
        self.epsilon = 1.0
        self.first_loop = True
        self.cycle_start = None
        self.cycle_type = None
        self.pause_end = None
        self.low_power_start_time = None
        self.prev_hp_power = 0.0
        self.prev_flow_temp = 0.0
        self.prev_cop = 3.5
        self.prev_actual_loss = 0.0
        self.undetected_count = 0
        self.prev_time = datetime.now(timezone.utc)

# Actor-Critic model (unchanged for Phase 2)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# Sim step function (vectorized demand calcs)
def sim_step(state, config, model, optimizer):
    try:
        current_time = datetime.now(timezone.utc)
        if (current_time - state.prev_time).total_seconds() < 60:
            time.sleep(60)
            return state.action_counter, state.prev_flow, state.prev_mode, state.prev_demand

        # Fetch entities with safe_float
        soc = safe_float(fetch_ha_entity(config['entities']['battery_soc']))
        solar_production = safe_float(fetch_ha_entity(config['entities']['solar_production']))
        grid_power = safe_float(fetch_ha_entity(config['entities']['grid_power']))
        hp_output = safe_float(fetch_ha_entity(config['entities']['hp_output']))
        hp_energy_rate = safe_float(fetch_ha_entity(config['entities']['hp_energy_rate']))
        total_heating_energy = safe_float(fetch_ha_entity(config['entities']['total_heating_energy']))
        ext_temp = safe_float(fetch_ha_entity(config['entities']['outdoor_temp']))
        forecast_min_temp = safe_float(fetch_ha_entity(config['entities']['forecast_weather'], 'temperature_low'))
        forecast_max_wind = safe_float(fetch_ha_entity(config['entities']['forecast_weather'], 'wind_speed'))
        delta_t = safe_float(fetch_ha_entity(config['entities']['primary_diff']))
        current_flow_temp = safe_float(fetch_ha_entity(config['entities']['hp_flow_temp']))
        hp_power = hp_energy_rate / 1000.0
        live_cop = hp_output / hp_power if hp_power > 0 else 3.5  # Fallback COP
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        hot_water_active = fetch_ha_entity(config['entities']['water_heater']) == 'heat'
        flow_min = safe_float(fetch_ha_entity(config['entities']['flow_min_temp']), default=25.0)
        flow_max = safe_float(fetch_ha_entity(config['entities']['flow_max_temp']), default=55.0)

        # Vectorized demand calculations
        rooms_list = list(config['rooms'].keys())
        areas = torch.tensor([config['rooms'][r] for r in rooms_list], dtype=torch.float32)
        facings = torch.tensor([config['facings'].get(r, 0.5) for r in rooms_list], dtype=torch.float32)
        emitter_kws = torch.tensor([config['emitter_kw'].get(r, 1.0) for r in rooms_list], dtype=torch.float32)
        room_temps = {r: safe_float(fetch_ha_entity(config['entities'][config['zone_sensor_map'].get(r, 'independent_sensor01')])) for r in rooms_list}
        room_temps_vec = torch.tensor([room_temps[r] for r in rooms_list], dtype=torch.float32)
        room_targets = torch.tensor([safe_float(fetch_ha_entity(config['entities'].get(f'{r}_heating'), 'target_temp')) for r in rooms_list], dtype=torch.float32)
        delta_temps = room_targets - room_temps_vec

        maintenance = areas * (config['peak_loss'] / (config['design_target'] - config['peak_ext'])) * (room_targets - ext_temp) / 1000 * facings
        maintenance_loss = torch.sum(maintenance).item()

        deficit = torch.clamp(delta_temps, min=0) * emitter_kws * 0.1
        deficit_loss = torch.sum(deficit).item()

        heat_up = torch.clamp(delta_temps, min=0) * areas * config['thermal_mass_per_m2'] / (config['heat_up_tau_h'] * 3600 / 1000)
        aggregate_heat_up = torch.sum(heat_up).item()

        # Other calcs remain looped or scalar as needed
        # ... (rest of sim_step logic unchanged, but replace loop-based maintenance/deficit/heat_up with vector sums)

        # Reward using coeffs
        reward = REWARD_COEFFS['rate_pen'] * current_rate * total_demand_adjusted / live_cop if live_cop > 0 else 0.0
        reward += REWARD_COEFFS['cop_bonus'] * (live_cop - 3.0) + REWARD_COEFFS['heat_up_pen'] * abs(aggregate_heat_up)
        # ... (apply to all reward terms)

        # ... (rest unchanged)

        state.prev_time = current_time
        return state.action_counter + 1, optimal_flow, optimal_mode, total_demand_adjusted

    except Exception as e:
        logging.error(f"Sim step error: {e}")
        state.prev_time = current_time
        return state.action_counter + 1, state.prev_flow, state.prev_mode, state.prev_demand

state_dim = 14
action_dim = 2
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
states = torch.zeros(state_dim)
qsh_state = QSHState()

def live_loop(qsh_state, states, config, model, optimizer):
    while True:
        qsh_state.action_counter, qsh_state.prev_flow, qsh_state.prev_mode, qsh_state.prev_demand = sim_step(qsh_state, states, config, model, optimizer)
        time.sleep(120)

live_loop(qsh_state, states, HOUSE_CONFIG, model, optimizer)