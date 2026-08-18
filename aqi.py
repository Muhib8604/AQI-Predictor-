import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Priority order:
# 1. Zafar Memon DHA
# 2. Karachi Saddar Town
#
# These are the CURRENT AQICN station IDs.
AQICN_STATIONS = [
    "A545140",      # Zafar Memon DHA
    "A471613",     # Karachi Saddar Town
]


def get_aqicn_data(station_id: str = None) -> dict | None:
    aqicn_api_key = os.getenv("AQICN_API_KEY")

    if not aqicn_api_key:
        print("ERROR: AQICN_API_KEY environment variable is missing.")
        return None

    # If a specific station is supplied, only try that one.
    # Otherwise try stations in priority order.
    stations_to_try = [station_id] if station_id else AQICN_STATIONS

    for sid in stations_to_try:

        sid = str(sid).strip().lstrip("@")

        # IMPORTANT:
        # AQICN station API uses the station ID directly.
        # Do NOT add '@' before the ID.
        url = f"https://api.waqi.info/feed/{sid}/"

        params = {
            "token": aqicn_api_key
        }

        try:
            print(f"Trying AQICN station: {sid}")

            response = requests.get(
                url,
                params=params,
                timeout=15
            )

            print(f"  HTTP status: {response.status_code}")

            response.raise_for_status()

            data = response.json()

            # AQICN normally returns:
            # {
            #   "status": "ok",
            #   "data": {...}
            # }
            status = data.get("status")

            if status != "ok":
                print(
                    f"  Station {sid} returned AQICN status: "
                    f"{status}"
                )

                # Print the actual API error when available.
                if "data" in data:
                    print(f"  AQICN response data: {data['data']}")

                continue

            payload = data.get("data")

            if not isinstance(payload, dict):
                print(
                    f"  Station {sid} returned invalid/missing "
                    f"data payload."
                )
                continue

            aqi = payload.get("aqi")

            # AQICN can sometimes return "-" instead of a numeric AQI.
            if aqi is None or aqi == "-":
                print(
                    f"  Station {sid} responded successfully "
                    f"but has no usable AQI value."
                )
                continue

            try:
                aqi = float(aqi)
            except (TypeError, ValueError):
                print(
                    f"  Station {sid} returned a non-numeric "
                    f"AQI value: {aqi}"
                )
                continue

            station_name = (
                payload
                .get("city", {})
                .get("name", f"Station {sid}")
            )

            print(
                f"  SUCCESS → Station: {station_name} "
                f"| AQI: {aqi}"
            )

            iaqi = payload.get("iaqi", {})

            return {
                "station_id": sid,
                "station_name": station_name,
                "aqi": aqi,

                "pm1": iaqi.get("pm1", {}).get("v"),
                "pm25": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),

                "aqi_temperature": iaqi.get("t", {}).get("v"),
                "aqi_humidity": iaqi.get("h", {}).get("v"),

                "timestamp": (
                    payload
                    .get("time", {})
                    .get("iso")
                ),
            }

        except requests.exceptions.Timeout:
            print(
                f"  Station {sid} timed out."
            )
            continue

        except requests.exceptions.RequestException as e:
            print(
                f"  Station {sid} HTTP/request error: {e}"
            )
            continue

        except ValueError as e:
            print(
                f"  Station {sid} returned invalid JSON: {e}"
            )
            continue

        except Exception as e:
            print(
                f"  Station {sid} failed unexpectedly: {e}"
            )
            continue

    print(
        "All AQICN stations failed or returned "
        "no usable AQI."
    )

    return None


if __name__ == "__main__":
    result = get_aqicn_data()
    print("\nFinal AQICN result:")
    print(result)