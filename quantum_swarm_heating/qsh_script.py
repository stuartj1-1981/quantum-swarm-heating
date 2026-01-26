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
    if isinstance(entity_id, list):
        for eid in entity_id:
            data_single = data.copy()
            data_single['entity_id'] = eid
            try:
                r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data_single)
                r.raise_for_status()
            except Exception as e:
                logging.error(f"HA set error for {eid}: {e}")
    else:
        try:
            r = requests.post(f"{HA_URL}/services/{domain}/{service}", headers=headers, json=data)
            r.raise_for_status()
        except Exception as e:
            logging.error(f"HA set error for {entity_id or data.get('device_id')}: {e}")

# HOUSE_CONFIG
HOUSE_CONFIG = {
    'rooms': { 'lounge': 19.48, 'open_plan': 42.14, 'utility': 3.40, 'cloaks': 2.51,
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
        'hall': 9.15, 'landing': 10.09 },
    'facings': { 'lounge': 0.2, 'open_plan': 1.0, 'utility': 0.5, 'cloaks': 0.5,
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
        'hall': 0.2, 'landing': 0.2 },
    'entities': {
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
        # Added for Tado valve % open (adjust IDs based on your HA setup; these are examples from common Tado naming)
        'lounge_heating': 'sensor.lounge_heating',
        'open_plan_heating': 'sensor.living_area_heating',  # Note: For list, average or use primary
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
        'landing_heating': 'sensor.landing_heating',
    },
    'zone_sensor_map': { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan': 'independent_sensor04',
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' },
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
    'peak_loss': 5.0,
    'design_target': 21.0,
    'peak_ext': -3.0,
    'thermal_mass_per_m2': 0.03,
    'heat_up_tau_h': 1.0,
    'persistent_zones': ['bathroom', 'ensuite1', 'ensuite2'],
    'hp_flow_service': {
        'domain': 'octopus_energy',
        'service': 'set_heat_pump_flow_temp_config',
        'device_id': 'b680894cd18521f7c706f1305b7333ea',
        'base_data': {
            'weather_comp_enabled': False
        }
    },
    'hp_hvac_service': {
        'domain': 'climate',
        'service': 'set_hvac_mode',
        'device_id': 'b680894cd18521f7c706f1305b7333ea'
    },
    # Default room_control_mode (overridable via options.json)
    'room_control_mode': {
        'lounge': 'indirect',  # Starter: Test one new TRV in direct
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
}

# Override room_control_mode from options.json if present
if 'room_control_mode' in user_options:
    HOUSE_CONFIG['room_control_mode'] = user_options['room_control_mode']
    logging.info(f"Overrode room_control_mode from options.json: {HOUSE_CONFIG['room_control_mode']}")

# Zone offsets inferred from logs for balanced heating
ZONE_OFFSETS = {
    'lounge': 0.1,
    'open_plan': -0.1,
    'utility': 0.1,
    'cloaks': 0.1,
    'bed1': 0.6,
    'bed2': 0.6,
    'bed3': 0.4,
    'bed4': 0.4,
    'bathroom': 0.0,  # Persistent, so offset not added to 25
    'ensuite1': 0.0,
    'ensuite2': 0.0,
    'hall': 0.1,
    'landing': 0.4
}

# Cycle detection constants - updated thresholds for boosted sensitivity
MIN_MODULATION_POWER = 0.20
LOW_POWER_MIN_IGNORE = 180  # Reduced to 180s
LOW_POWER_MAX_TOLERANCE = 1800
FLOW_TEMP_SPIKE_THRESHOLD = 1.5  # Lowered
POWER_SPIKE_THRESHOLD = 0.3  # Lowered
FLOW_TEMP_DROP_THRESHOLD = -1.5  # Lowered abs for sensitivity
COP_DROP_THRESHOLD = -0.3  # Lowered
COP_SPIKE_THRESHOLD = 0.3  # Lowered
DEMAND_DELTA_THRESHOLD = 1.5  # Added for Δdemand
LOSS_DELTA_THRESHOLD = 0.5  # Added for loss spikes (kW)
EXTENDED_RECOVERY_TIME = 300  # For extended pauses

# New globals for per-room nudges/drops (hysteresis, cooldown)
room_nudge_hyst = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_cooldown = {room: 0 for room in HOUSE_CONFIG['rooms']}
room_nudge_accum = {room: 0.0 for room in HOUSE_CONFIG['rooms']}  # Accumulated nudges for clamp

# Global for persisted rates
prev_all_rates = []

def parse_rates_array(rates_list, suppress_warning=False):
    if not rates_list or not isinstance(rates_list, list) or len(rates_list) == 0:
        if not suppress_warning:
            logging.warning("Rates list empty or invalid—using fallback rates.")
        return []
    try:
        return [(r['start'], r['end'], r['value_inc_vat']) for r in rates_list if 'start' in r and 'end' in r and 'value_inc_vat' in r]
    except Exception as e:
        logging.error(f"Rate parse error: {e} — using fallback rates.")
        return []

def get_current_rate(rates):
    now = datetime.now(timezone.utc)
    for start, end, price in rates:
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            if start_dt <= now < end_dt:
                return price
        except ValueError as e:
            logging.warning(f"Invalid date in rates: {e} — skipping entry.")
    return HOUSE_CONFIG['fallback_rates']['standard']

def calc_solar_gain(config, production):
    return production * 0.5

def calc_room_loss(config, room, delta_temp, chill_factor=1.0, loss_coeff=0.0, sum_af=0.0):
    area = config['rooms'].get(room, 0)
    facing = config['facings'].get(room, 1.0)
    if sum_af == 0:
        return 0.0
    room_coeff = (area * facing / sum_af) * loss_coeff
    loss = room_coeff * max(0, delta_temp) * chill_factor
    return loss

def total_loss(config, ext_temp, room_targets, chill_factor=1.0, loss_coeff=0.0, sum_af=0.0):
    return sum(calc_room_loss(config, room, room_targets[room] - ext_temp, chill_factor, loss_coeff, sum_af) for room in config['rooms'])

def build_dfan_graph(config):
    G = nx.Graph()
    for room in config['rooms']:
        G.add_node(room, area=config['rooms'][room], facing=config['facings'][room])
    G.add_edges_from([('lounge', 'hall'), ('open_plan', 'utility')])
    return G

class SimpleQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        return self.fc(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = SimpleQNet(state_dim, action_dim)
        self.critic = SimpleQNet(state_dim, 1)

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value

def train_rl(model, optimizer, episodes=500):
    for _ in range(episodes):
        # Simulate hypothetical state (random for variety)
        sim_state = torch.rand(13) * 10
        action, value = model(sim_state.unsqueeze(0))  # <--- Critical: This line defines 'action' and 'value'
        action = action.squeeze(0)  # <--- Squeeze the batch dim (from (1, action_dim) to (action_dim,))
        
        # Simulate det logic approx (e.g., high demand → heat, high flow)
        sim_demand = sim_state[4].item()  # Index 4: smoothed_demand
        sim_mode = 1.0 if sim_demand > 5 else 0.0
        sim_flow = 45.0 if sim_demand > 5 else 35.0
        norm_flow = (sim_flow - 30) / 20
        
        # Simulate reward (good if matches sim)
        mode_match = (F.sigmoid(action[0]) > 0.5) == (sim_mode == 1.0)
        flow_err = abs(action[1].item() - norm_flow)
        sim_reward = 1.0 if mode_match and flow_err < 0.1 else -1.0
        
        td_error = sim_reward - value.item()
        critic_loss = td_error ** 2
        
        mode_log_prob = F.binary_cross_entropy_with_logits(action[0], torch.tensor(sim_mode))
        flow_mse = (action[1] - torch.tensor(norm_flow)).pow(2)
        actor_loss = mode_log_prob + flow_mse
        
        loss = critic_loss + 0.5 * actor_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Initial RL training complete with simulated scenarios.")

# Globals for refinements
demand_history = deque(maxlen=5)  # Extended to 5
prod_history = deque(maxlen=5)
grid_history = deque(maxlen=5)
cop_history = deque([4.0] * 5, maxlen=5)  # For median non-zero COP, init with Cosy 6 mild baseline
heat_up_history = deque(maxlen=5)  # (time, aggregate_heat_up) for rate
low_delta_persist = 0
low_power_start_time = None
prev_hp_power = 1.0  # Better init: assume moderate power
prev_flow_temp = 35.0  # Better init: typical flow
prev_cop = 3.5
cycle_type = None
cycle_start = None
pause_end = None  # New for fixed pause duration
action_counter = 0
prev_flow = 35.0
prev_mode = 'off'
prev_demand = 3.5  # Better init: fallback demand
prev_time = time.time()
prev_actual_loss = 0.0
reward_history = deque(maxlen=1000)
loss_history = deque(maxlen=1000)
pause_count = 0
undetected_count = 0
enable_plots = user_options.get('enable_plots', False)
first_loop = True  # New: Flag for startup
epsilon = 0.2  # Initial exploration rate
blend_factor = 0.0  # Start at 0 for stability testing; ramp later
last_heat_time = time.time() - 600  # Init as expired
consecutive_slow = 0  # For heat_up hysteresis

def shutdown_handler(sig, frame):
    mean_reward = sum(reward_history) / len(reward_history) if reward_history else 0
    mean_loss = sum(loss_history) / len(loss_history) if loss_history else 0
    demand_list = list(demand_history)
    demand_std = np.std(demand_list) if demand_list else 0
    logging.info(f"Shutdown summary: mean_reward={mean_reward:.2f}, mean_loss={mean_loss:.2f}, pause_count={pause_count}, demand_std={demand_std:.2f}, undetected_events={undetected_count}")
    if enable_plots and demand_list:
        import matplotlib.pyplot as plt  # Import here to avoid ModuleNotFoundError if not enabled
        plt.plot(demand_list)
        plt.title('Demand History')
        plt.xlabel('Steps')
        plt.ylabel('Demand (kW)')
        plt.savefig('/data/demand_hist.png')
        logging.info("Demand history plot saved to /data/demand_hist.png")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

def sim_step(graph, states, config, model, optimizer, action_counter, prev_flow, prev_mode, prev_demand):
    global low_delta_persist, low_power_start_time, prev_hp_power, prev_flow_temp, prev_cop, cycle_type, demand_history, prod_history, grid_history, prev_time, cycle_start, prev_actual_loss, pause_count, undetected_count, first_loop, pause_end, prev_all_rates, epsilon, blend_factor, cop_history, heat_up_history, last_heat_time, consecutive_slow, room_nudge_hyst, room_nudge_cooldown, room_nudge_accum
    try:
        current_time = time.time()
        time_delta = current_time - prev_time
        dfan_control = fetch_ha_entity(config['entities']['dfan_control_toggle']) == 'on'
        target_temp = float(fetch_ha_entity(config['entities']['pid_target_temperature']) or 21.0)
        logging.info(f"Using target_temp: {target_temp}°C from pid_target_temperature.")
        
        # Fetch hot_water_active early to decide on pause
        hot_water_active = fetch_ha_entity(config['entities']['water_heater']) == 'high_demand'
        
        # Compute room_targets (independent, for shadow logging only)
        room_targets = {room: target_temp + ZONE_OFFSETS.get(room, 0.0) for room in config['rooms']}
        for room in config['persistent_zones']:
            room_targets[room] = 25.0
        
        # Minimal fetching for log
        ext_temp = float(fetch_ha_entity(config['entities']['outdoor_temp']) or 0.0)
        current_day_rates = parse_rates_array(fetch_ha_entity(config['entities']['current_day_rates'], 'rates') or [], suppress_warning=(datetime.now(timezone.utc).hour < 16))
        current_rate = get_current_rate(current_day_rates)
        
        if hot_water_active:
            logging.info("Hot water active: Pausing all QSH processing (space heating optimizations, RL updates, cycle detection, and HA sets).")
            optimal_mode = 'off'
            optimal_flow = prev_flow  # Retain previous to avoid jumps on resume
            total_demand_adjusted = 0.0
            
            # Minimal logging for mode decision (demand=0)
            logging.info(f"Mode decision: optimal_mode='off', total_demand=0.00 kW, ext_temp={ext_temp:.1f}°C, upcoming_cold=False, upcoming_high_wind=False, current_rate={current_rate:.3f} GBP/kWh, hot_water_active=True")
            
            # Reset cycle/low-power states to avoid false positives on resume
            low_delta_persist = 0
            low_power_start_time = None
            cycle_type = None
            cycle_start = None
            pause_end = None
            
            # Set defaults for other variables to avoid undefined errors
            smoothed_demand = 0.0
            demand_std = 0.0
            chill_factor = 1.0
            smoothed_grid = 0.0
            forecast_min_temp = 0.0
            wind_speed = 0.0
            excess_solar = 0.0
            soc = 50.0
            live_cop = prev_cop
            delta_t = 3.0
            hp_power = float(fetch_ha_entity(config['entities']['hp_energy_rate']) or 0.0)
            upcoming_cold = False
            upcoming_high_wind = False
            actual_loss = 0.0
            sum_af = 0.0
            flow_min = float(fetch_ha_entity(config['entities']['flow_min_temp']) or 32.0)
            flow_max = float(fetch_ha_entity(config['entities']['flow_max_temp']) or 50.0)
            current_flow_temp = float(fetch_ha_entity(config['entities']['hp_flow_temp']) or 35.0)
            reward = 0.0  # Default, since no RL update
            loss = torch.tensor(0.0)  # Default
            heat_up_power = 0.0
            export_kw = 0.0
            
            # Skip to shadow sets and return
            clamped_demand = max(min(total_demand_adjusted, 15.0), 0.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_total_demand', 'value': clamped_demand})
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_shadow_flow', 'value': optimal_flow})
            set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_shadow_mode', 'option': optimal_mode})
            set_ha_service('input_select', 'select_option', {'entity_id': 'input_select.qsh_optimal_mode', 'option': optimal_mode})
            clamped_reward = max(min(reward, 100.0), -100.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_reward', 'value': clamped_reward})
            clamped_loss = max(min(loss.item(), 2000.0), 0.0)
            set_ha_service('input_number', 'set_value', {'entity_id': 'input_number.qsh_rl_loss', 'value': clamped_loss})

            for room in config['rooms']:
                shadow_setpoint = room_targets[room]
                entity_id = f'input_number.qsh_shadow_{room}_setpoint'
                set_ha_service('input_number', 'set_value', {'entity_id': entity_id, 'value': shadow_setpoint})
                if not dfan_control:
                    logging.info(f"Shadow: Would set {room} to {shadow_setpoint:.1f}°C")

            prev_time = current_time
            return action_counter + 1, optimal_flow, optimal_mode, total_demand_adjusted
        
        # Else: Normal processing
        # Fetch sensor temps
        sensor_temps = {
            'independent_sensor01': float(fetch_ha_entity(config['entities']['independent_sensor01']) or target_temp),
            'independent_sensor02': float(fetch_ha_entity(config['entities']['independent_sensor02']) or target_temp),
            'independent_sensor03': float(fetch_ha_entity(config['entities']['independent_sensor03']) or target_temp),
            'independent_sensor04': float(fetch_ha_entity(config['entities']['independent_sensor04']) or target_temp),
        }
        current_temps = {room: sensor_temps[config['zone_sensor_map'][room]] for room in config['rooms']}

        # Fetch forecast
        forecast = fetch_ha_entity(config['entities']['forecast_weather'], 'forecast') or []
        forecast_temps = [f['temperature'] for f in forecast if 'temperature' in f]
        forecast_wind_speeds = [f['wind_speed'] for f in forecast if 'wind_speed' in f]
        forecast_min_temp = min(forecast_temps or [ext_temp])
        wind_speed = float(fetch_ha_entity(config['entities']['forecast_weather'], 'wind_speed') or 0.0)
        upcoming_cold = any(t < 0 for t in forecast_temps)
        upcoming_high_wind = any(w > 15 for w in forecast_wind_speeds)
        chill_factor = 1.0 + (wind_speed / 10.0) * 0.1

        # Fetch solar, grid, soc, hp_output
        solar_production = float(fetch_ha_entity(config['entities']['solar_production']) or 0.