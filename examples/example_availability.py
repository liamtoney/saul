from saul import Stream

df = Stream.from_earthscope(
    'AK,AV',
    'HOM,RC01,BAEI',
    '?DF,BHZ',
    (2024, 8, 1),
    (2025, 2, 1),
    just_availability=True,
)
