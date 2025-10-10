from tqdm import tqdm
import time, math

with tqdm(range(100000), desc="Ottimizzazione", unit="iter") as t:
    for i in t:
        cost = math.exp(-i / 20000)
        t.set_postfix({"Costo": f"{cost:.6f}"})
        time.sleep(0.001)
