from datetime import datetime, timezone, timedelta
import json
import logging
import numpy as np   # if used for std etc.

def safe_float(value, default=0.0):
    try:
        if value in (None, "", "unknown", "unavailable"):
            return default
        return float(value)
    except (TypeError, ValueError):
        logging.warning(f"safe_float: could not parse '{value}', using default {default}")
        return default

def safe_int(value, default=0):
    try:
        if value in (None, "", "unknown", "unavailable"):
            return default
        return int(value)
    except (TypeError, ValueError):
        logging.warning(f"safe_int: could not parse '{value}', using default {default}")
        return default

def safe_json(value, default=None):
    if default is None:
        default = {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed
        except json.JSONDecodeError:
            logging.warning(f"safe_json: could not parse JSON from '{value}', using default")
            return default
    return default

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

def shutdown_handler(sig, frame):
    mean_reward = sum(reward_history) / len(reward_history) if reward_history else 0
    mean_loss = sum(loss_history) / len(loss_history) if loss_history else 0
    demand_list = list(demand_history)
    demand_std = np.std(demand_list) if demand_list else 0
    logging.info(f"Shutdown summary: mean_reward={mean_reward:.2f}, mean_loss={mean_loss:.2f}, pause_count={pause_count}, demand_std={demand_std:.2f}, undetected_events={undetected_count}")
    if enable_plots and demand_list:
        try:
            import matplotlib.pyplot as plt
            plt.plot(demand_list)
            plt.title('Demand History')
            plt.xlabel('Steps')
            plt.ylabel('Demand (kW)')
            plt.savefig('/data/demand_hist.png')
            logging.info("Demand history plot saved to /data/demand_hist.png")
        except ImportError:
            logging.warning("matplotlib not installed; skipping plot generation.")
    sys.exit(0)

def get_current_temp(room, sensor_temps, target_temp=21.0):
    sensor_key = HOUSE_CONFIG['zone_sensor_map'].get(room, 'independent_sensor01')
    return safe_float(sensor_temps.get(sensor_key, target_temp), target_temp)