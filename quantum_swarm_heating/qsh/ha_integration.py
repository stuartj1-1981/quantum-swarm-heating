HA_URL = 'http://supervisor/core'
TOKEN = os.getenv('SUPERVISOR_TOKEN')
REQUEST_TIMEOUT = 2  # seconds

headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'} if TOKEN else None

def fetch_ha_entity(entity_id, attr=None, default=None):
    if not headers:
        logging.warning("No SUPERVISOR_TOKEN found—fetch_ha_entity will return default.")
        return default
    try:
        response = requests.get(
            f'{HA_URL}/api/states/{entity_id}',
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        if attr:
            return data.get('attributes', {}).get(attr, default)
        return data.get('state', default)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching entity {entity_id}: {e}")
        return default
    except (ValueError, TypeError) as e:
        logging.error(f"Error parsing response for entity {entity_id}: {e}")
        return default

def set_ha_service(domain, service, data):
    if not headers:
        logging.warning("No SUPERVISOR_TOKEN found—set_ha_service skipped.")
        return
    try:
        response = requests.post(
            f'{HA_URL}/api/services/{domain}/{service}',
            json=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        logging.info(f"Service {domain}.{service} called successfully.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling service {domain}.{service}: {e}")