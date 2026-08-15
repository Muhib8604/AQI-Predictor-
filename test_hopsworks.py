import traceback
import hopsworks

print("Version:", hopsworks.__version__)

try:
    project = hopsworks.login()
    print("SUCCESS")
except Exception:
    traceback.print_exc()