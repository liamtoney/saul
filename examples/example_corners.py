from obspy import read_inventory

from saul import calculate_responses

# Read in default inventory
inv = read_inventory()

# Calculate sensor responses and corner frequencies
df = calculate_responses(inv, sampling_rate=10, plot=True)
