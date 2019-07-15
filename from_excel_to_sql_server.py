"""
@time: 6/26/2019 7:35 PM

@author: 柚子
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('mssql+pymssql://localhost:1433/master',echo=True)
dbsession = sessionmaker(engine)
session = dbsession()

def reshape_df_cols(df):
    cols = df.columns
    reshape_cols = [col.replace("(", "_").replace(")", "") for col in cols]
    df.columns = reshape_cols
    return df


def insert_data(excel_file_name, db_table_name,date_cols, schema="raw",idx=True,rename=None):
    data = pd.read_excel(os.path.join(raw_file, excel_file_name))
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], format="%Y-%m-%d")
    data = reshape_df_cols(data)
    df_len,col_num = data.shape
    data = data.iloc[:,:col_num+1]
    if rename:
        data.rename(columns=rename, inplace=True)
    data.to_sql(name=db_table_name, con=engine, schema=schema, index=idx, if_exists="append")


def delete_from_raw_table(table_name,schema="raw"):
    session.execute(f"delete from [{schema}].[{table_name}]")


if __name__ == '__main__':

    raw_file = r'C:\Users\Ruofei Shen\Desktop\mck_studies\wistron\yibai0626\input data templates\frontend data templates'
    raw_to_db_names_mapping = [["yield rate.xlsx", "yield rate",["Month"]],
                            ["R3M p&l.xlsx", "r3m p&l",[]],
                            ["project info.xlsx", "project info",["MP Start", "Shipment Start"]],
                            ["budget p&l.xlsx", "budget p&l",[]],
                            ["Actual p&l for loading history only.xlsx", "actual p&l",[]]
                            ["FX rate.xlsx", "FX rate",["Month"]],                      # for actual fx rate
                            ["budget_rate.xlsx", "Budget_fx", []],                # for budget rate
                            ["booking_rate.xlsx", "R3M_fx", ["Month"]],                 # for r3m rate
                            ["Scrap amount.xlsx", "Scrap amount",["Month"]],
                            ["input owner and deadline.xlsx", "input owner and deadline",[]],
                            ["ADS.xlsx", "ADS",["Month"]],
                            ["APR.xlsx", "APR",["Month", "Version date"]],
                            ["yield rate.xlsx", "yield rate",["Month"]],
                            ["accounting item mapping.xlsx", "accounting item mapping",[]],         # map of accouting_item_code and accounting_item_desc
                            ["actual p&l to standard line item names mapping.xlsx", "raw_line_items_map",[]],   # map from raw accouting item names to accouting_item_desc in db
                            ["health_score_weights.xlsx", "health_score_weights",[]],       # weights of 6 dimensions for counting weighted average health score
                            ["line_item_to_app_item.xlsx", "line_item_merge_map",[]],       # 在计算app.perf_overview_model科目需要合并
                            ["param_control.xlsx", "param_control",["date"]]]
    for excel_file_name, db_table_name, date_cols in raw_to_db_names_mapping:
        insert_data(excel_file_name, db_table_name,date_cols)

    excel_file_name, db_table_name, date_cols = ["SDBG cost item list.xlsx","tbl_cost_control_list",[]]
    insert_data(excel_file_name, db_table_name,date_cols,schema="dim",idx=False)
    excel_file_name, db_table_name, date_cols = ["app_cost_category.xlsx","cost_category",[]]   # 2019.7.5 luke's app table
    insert_data(excel_file_name, db_table_name, date_cols, schema="app", idx=False)