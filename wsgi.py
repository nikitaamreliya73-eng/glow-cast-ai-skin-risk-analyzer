import importlib.util
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = os.path.join(BASE_DIR, "app.py")

spec = importlib.util.spec_from_file_location("glowcast_main", APP_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app
