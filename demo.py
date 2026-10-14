import sys
from hw2_task1 import run_session
 
if len(sys.argv) < 2 or sys.argv[1] not in ("a", "b"):
    sys.exit("Usage: python demo.py a|b")
 
if sys.argv[1] == "a":
    run_session("Check the Silverbox error.", "run_a")
else:
    run_session("Run the Silverbox PINN script and report the test RMSE.", "run_b")
 