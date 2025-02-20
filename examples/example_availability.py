from saul import get_availability

df = get_availability(
    'AK,AV',
    'HOM,RC01,BAEI',
    '?DF,BHZ',
    (2024, 8, 1),
    (2025, 2, 1),
    print_timespans=True,
    plot=True,
)
