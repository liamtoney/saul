from saul import Stream

st = Stream.from_earthscope(
    'UU', 'ZNPU,KNB,LCMT', 'HHN', (2023, 11, 14, 22, 38), (2023, 11, 14, 22, 40)
)
st.detrend('demean').taper(0.05).remove_response(output='DISP')
st.taper(0.05).filter('highpass', freq=2)
st.plot(
    src_lat=37.26842,
    src_lon=-112.93468,
    reftime=(2023, 11, 14, 22, 38, 42),
    wavespeed=2.6,  # [km/s]
)
