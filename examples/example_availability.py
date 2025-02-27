from saul import get_availability

df = get_availability(
    'AK,AV',
    'HOM,RC01,KENI',
    '?DF,BHZ',
    (2024, 8, 1),
    (2024, 10, 1),
    print_timespans=True,
    plot=True,
)
