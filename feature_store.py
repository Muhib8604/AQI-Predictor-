import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def connect_feature_store():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
        cert_folder=r"E:\AQI Predictor\hops_certs",
    )

    return project.get_feature_store()