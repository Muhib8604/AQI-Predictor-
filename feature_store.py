import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def connect_feature_store():
    cert_folder = os.path.join(os.getcwd(), "hops_certs")
    os.makedirs(cert_folder, exist_ok=True)

    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
        cert_folder=cert_folder,
    )

    return project.get_feature_store()