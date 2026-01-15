import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime, timedelta
import time
import logging
import os
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load user options
try:
    with open('/data/options.json', 'r') as f:
        user_options = json.load(f)
except Exception as e:
    logging.warning(f"Failed to load options.json: {e}. Using defaults.")
    user_options = {}

logging.info(f"Loaded user_options: {user_options}")  # Debug

# HA API setup (pull token from options or env)
HA_URL = os.getenv('HA_URL', 'http://supervisor/core/api')
HA_TOKEN = user_options.get('ha_token') or os.getenv('HA_TOKEN')

logging.info(f"Detected HA_TOKEN: {'Set' if HA_TOKEN else 'None'}")  # Debug (hides actual token for security)

if not HA_TOKEN:
    logging.critical("HA_TOKEN not set! Using defaults only. Check add-on config or env.")
else:
    logging.info("HA_TOKEN found—using real HA API calls.")

def fetch_ha_entity(entity_id, attr=None):
    if not HA_TOKEN:
        return None
    headers = {'Authorization': f"Bearer {HA_TOKEN}"}
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

# Default config
HOUSE_CONFIG = {
    'rooms': { 'lounge': 19.48, 'open_plan_ground': 42.14, 'utility': 3.40, 'cloaks': 2.51,
        'bed1': 18.17, 'bed2': 13.59, 'bed3': 11.07, 'bed4': 9.79, 'bathroom': 6.02, 'ensuite1': 6.38, 'ensuite2': 3.71,
        'hall': 9.15, 'landing': 10.09 },
    'facings': { 'lounge': 0.2, 'open_plan_ground': 1.0, 'utility': 0.5, 'cloaks': 0.5,
        'bed1': 0.2, 'bed2': 1.0, 'bed3': 0.5, 'bed4': 0.5, 'bathroom': 0.2, 'ensuite1': 0.5, 'ensuite2': 1.0,
        'hall': 0.2, 'landing': 0.2 },
    'entities': {
        'lounge_temp_set_hum': 'climate.tado_smart_radiator_thermostat_va4240580352',
        'open_plan_ground_temp_set_hum': ['climate.tado_smart_radiator_thermostat_va0349246464', 'climate.tado_smart_radiator_thermostat_va3553629184'],
        'independent_sensor01': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor01_temperature',
        'independent_sensor02': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor02_temperature',
        'independent_sensor03': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor03_temperature',
        'independent_sensor04': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_sensor04_temperature',
        'battery_soc': 'sensor.givtcp_ce2029g082_soc',
        'battery_design_capacity_ah': 'sensor.givtcp_dx2327m548_battery_design_capacity',
        'battery_remaining_capacity_ah': 'sensor.givtcp_dx2327m548_battery_remaining_capacity',
        'battery_power': 'sensor.givtcp_ce2029g082_battery_power',
        'battery_voltage': 'sensor.givtcp_ba2027g052_battery_voltage_2',
        'ac_charge_power': 'sensor.givtcp_ce2029g082_ac_charge_power',
        'battery_to_grid': 'sensor.givtcp_ce2029g082_battery_to_grid',
        'battery_to_house': 'sensor.givtcp_ce2029g082_battery_to_house',
        'grid_voltage_2': 'sensor.givtcp_ce2029g082_grid_voltage_2',
        'grid_power': 'sensor.givtcp_ce2029g082_grid_power',
        'current_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_current_day_rates',
        'next_day_rates': 'event.octopus_energy_electricity_21l3885048_2700002762631_next_day_rates',
        'current_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_current_day_rates',
        'next_day_export_rates': 'event.octopus_energy_electricity_21l3885048_2700006856140_export_next_day_rates',
        'solar_production': 'sensor.envoy_122019031249_current_power_production',
        'outdoor_temp': 'sensor.front_door_temperature_measurement',
        'forecast_weather': 'weather.home',
        'hp_output': 'sensor.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31_live_heat_output',
        'hp_energy_rate': 'sensor.shellyem_c4d8d5001966_channel_1_power',
        'total_heating_energy': 'sensor.shellyem_c4d8d5001966_channel_1_energy',
        'hp_water_tonight': 'input_boolean.hp_chosen_for_tonight',
        'water_heater': 'water_heater.octopus_energy_heat_pump_00_1e_5e_09_02_b6_88_31',
        'flow_min_temp': 'input_number.flow_min_temperature',
        'flow_max_temp': 'input_number.flow_max_temperature',
        'hp_cop': 'sensor.live_cop_calc',
        'dfan_control_toggle': 'input_boolean.dfan_control'
    },
    'zone_sensor_map': { 'hall': 'independent_sensor01', 'bed1': 'independent_sensor02', 'landing': 'independent_sensor03', 'open_plan_ground': 'independent_sensor04',
        'utility': 'independent_sensor01', 'cloaks': 'independent_sensor01', 'bed2': 'independent_sensor02', 'bed3': 'independent_sensor03', 'bed4': 'independent_sensor03',
        'bathroom': 'independent_sensor03', 'ensuite1': 'independent_sensor02', 'ensuite2': 'independent_sensor03', 'lounge': 'independent_sensor01' },
    'hot_water': {'load_kw': 2.5, 'ext_threshold': 3.0, 'cycle_start_hour': 0, 'cycle_end_hour': 6, 'tank_low_threshold': 40.0},
    'battery': {'min_soc_reserve': 4.0, 'efficiency': 0.9, 'voltage': 51.0, 'max_rate': 3.0},
    'grid': {'nominal_voltage': 230.0, 'min_voltage': 200.0, 'max_voltage': 250.0},
    'fallback_rates': {'cheap': 0.1495, 'standard': 0.3048, 'peak': 0.4572, 'export': 0.15},
    'inverter': {'fallback_efficiency': 0.95},
    'peak_loss': 10.0,
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
    }
}

# Merge user options
if 'tado_rooms' in user_options and isinstance(user_options['tado_rooms'], list):
    HOUSE_CONFIG['entities'].update({item['room'] + '_temp_set_hum': item['entity'] for item in user_options['tado_rooms'] if isinstance(item, dict) and 'room' in item and 'entity' in item})
# Add similar for others as needed

def parse_rates_array(rates_str):
    if rates_str is None:
        return []
    try:
        rates = json.loads(rates_str) if isinstance(rates_str, str) else rates_str
        return [(r['start'], r['end'], r['value_inc_vat']) for r in rates.get('rates', [])]
    except Exception as e:
        logging.error(f"Rate parse error: {e}")
        return []

def get_current_rate(rates):
    now = datetime.now()
    for start, end, price in rates:
        if datetime.fromisoformat(start) <= now < datetime.fromisoformat(end):
            return price / 100
    return HOUSE_CONFIG['fallback_rates']['standard']

def calc_solar_gain(config, production):
    return production * 0.5

def calc_room_loss(config, room, delta_temp, chill_factor=1.0):
    area = config['rooms'].get(room, 0)
    facing = config['facings'].get(room, 1.0)
    loss = area * max(0, delta_temp) * facing * chill_factor / 10
    return loss

def total_loss(config, ext_temp, target_temp=21.0, chill_factor=1.0):
    delta = target_temp - ext_temp
    return sum(calc_room_loss(config, room, delta, chill_factor) for room in config['rooms'])

def build_dfan_graph(config):
    G = nx.Graph()
    for room in config['rooms']:
        G.add_node(room, area=config