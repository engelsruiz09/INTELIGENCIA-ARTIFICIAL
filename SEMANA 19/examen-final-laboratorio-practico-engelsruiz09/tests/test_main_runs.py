import subprocess

def test_main_runs():
    result = subprocess.run(["python", "src/main.py"], capture_output=True)
    assert result.returncode == 0, "main.py debe ejecutarse sin errores"
