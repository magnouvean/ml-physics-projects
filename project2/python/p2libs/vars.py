import os

current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

fig_directory = f"{os.path.dirname(os.path.dirname(current_dir))}/figures"
