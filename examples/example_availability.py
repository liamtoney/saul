from saul import Stream

df = Stream.from_earthscope(
    'AK,AV',
    'HOM,RC01,BAEI',
    '?DF,BHZ',
    (2023, 9, 1, 0, 5),
    (2023, 9, 1, 0, 15),
    just_availability=True,
)
