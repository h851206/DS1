import streamlit as st
import TingYu.measurement_table
import TingYu.Methods

st.title('DS1 T2-1')
st.text("Group members: TingYu")

red_wine_dir = '/Users/ting-yuho/Desktop/DS1/finalProject/redwine-quality.csv'
white_wine_dir = '/Users/ting-yuho/Desktop/DS1/finalProject/redwine-quality.csv'
raw_df = TingYu.Methods.combine_raw_data(data_red_dir=red_wine_dir, data_white_dir=white_wine_dir)
st.dataframe(raw_df)
