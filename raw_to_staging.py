"""
@time: 6/27/2019 11:00 AM

@author: 柚子
"""
import os
import re
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

raw_file = r'C:\Users\Ruofei Shen\Desktop\mck_studies\wistron\yibai0626\input data templates\frontend data templates'
engine = create_engine('mssql+pymssql://localhost:1433/master',echo=True)
dbsession = sessionmaker(engine)
session = dbsession()

def get_db_table_to_df(schema, db_table_name):
    data = pd.read_sql(f"select * from [{schema}].[{db_table_name}]", con=engine)
    return data


def find_site_in_project_info(data, left_key="Model"):
    # find model site by model name
    project_info = get_db_table_to_df("raw", "project info")
    data = pd.merge(data, project_info[['AllieCode Name', 'Site','Project name']], left_on=left_key, right_on=["Project name"],how="left")
    return data


def find_model_code_in_project_info(data, left_key="Model"):
    # find model code by model name
    project_info = get_db_table_to_df("raw", "project info")
    data = pd.merge(data, project_info[['AllieCode Name', "Project name"]], left_on=left_key, right_on=["Project name"],how="left",copy=False)
    return data


def find_r3m_version_date(data,left_key="reporting_month"):
    # 2019.7.l1 fx calculation changed, from one source to 3 sources
    r3m_raw = get_db_table_to_df("raw", "r3m p&l")
    reporting_month_version = r3m_raw[["Period", "Version"]].drop_duplicates()
    reporting_month_version["Period"] = pd.to_datetime(reporting_month_version["Period"], format="%Y/%m")
    reporting_month_version["Version"] = pd.to_datetime(reporting_month_version["Version"], format="%Y%m%d")
    reporting_month_version = reporting_month_version.set_index("Period")
    map_dict = reporting_month_version["Version"].to_dict()
    if data[left_key].dtype == 'O':
        data[left_key] = pd.to_datetime(data[left_key])
    data["Version"] = data[left_key].map(map_dict)
    return data


def unit_version_date_for_r3m(x):
    # 2019.7.l1 fx calculation changed, from one source to 3 sources
    x = pd.to_datetime(x)
    year = x.year
    month = x.month
    if month == 12:
        year += 1
        month = 1
    return pd.to_datetime(f"{year}-{month+1}-1")


def find_fx_rate(data, left_key="Month",type="Actual"):
    """
    :param data:df that need to find usd->NTD fx rate
    :param left_key: the key that stands for month time to merge with fx rate db table
    :param type: actual,r3m and budget get fx rate from different raw table (data should contain only one type)
    :return: merge fx rate to data
    # 2019.7.l1 fx calculation changed, from one source to 3 sources
    """
    if data[left_key].dtype == 'O':
        data["Month"] = pd.to_datetime(data[left_key])
        left_key = "Month"

    if type == "Actual":
        fx_rate = get_db_table_to_df("raw", "FX rate")
        fx_rate = fx_rate.set_index("Month")
    elif type == "Budget":
        data["Year"] = data[left_key].dt.year
        left_key = "Year"
        fx_rate = get_db_table_to_df("raw", "Budget_fx")
        fx_rate = fx_rate.set_index("Year")
    else:
        # R3M 需要根据version date查找rate，取version date的下一个月
        data["Month"] = data["Version"].apply(unit_version_date_for_r3m)
        fx_rate = get_db_table_to_df("raw", "R3M_fx")
        fx_rate = fx_rate.set_index("Month")
        left_key = "Month"
    fx_rate = fx_rate.query("(From=='USD')&(To=='NTD')")
    data = pd.merge(data, fx_rate['FX'],left_on=left_key, right_index=True, how="left", copy=False)
    return data


def find_fx_rate_with_type_col(data, left_key="reporting_month"):
    # 2019.7.l1 fx calculation changed, from one source to 3 sources
    # data contain multiple types, and should find fx seperately using function find_fx_rate
    init_cols = list(data.columns)
    init_cols.append("FX")
    type_list = data["type"].unique()
    result = pd.DataFrame()
    for type in type_list:
        one_type_data = data.query("type == @type")
        if type == "R3M":
            one_type_data = find_r3m_version_date(one_type_data, left_key=left_key)
        one_type_data = find_fx_rate(one_type_data, left_key=left_key, type=type)
        result = pd.concat([result, one_type_data[init_cols]])
    return result


def aligh_table_cols_with_db(db_table, df, remove_id=True):
    cols = list(db_table.columns)
    if remove_id:
        cols.remove("id")
    db_data=df[cols]
    return db_data


def raw_to_stage_indireact(db_tb_name, data2):
    """
    :param db_tb_name: staging table that data2 goes to
    :param data2:
    :return: use staging table cols to aquire data2's data and insert into staging table
    """
    tbl_kpi = pd.read_sql(f"select * from staging.[{db_tb_name}]",con=engine)
    db_data = aligh_table_cols_with_db(tbl_kpi,data2)
    db_data.to_sql(db_tb_name, con=engine, schema="staging",if_exists="append",index=False)


def direct_raw_to_stage(raw_tb_name, db_tb_name):
    raw_tb = pd.read_sql(f"select * from raw.[{raw_tb_name}]",con=engine)
    fx_db = pd.read_sql(f"select * from staging.[{db_tb_name}]",con=engine)
    data = aligh_table_cols_with_db(fx_db, raw_tb)
    data.to_sql(db_tb_name, con=engine, schema="staging",if_exists="append",index=False)


def find_accounting_item_desc(data):
    # pl_is_model_mapping = "actual p&l to standard line item names mapping.xlsx"
    # pl_is_model_map = pd.read_excel(os.path.join(raw_file, pl_is_model_mapping), sheet_name="Column header")
    pl_is_model_map = pd.read_sql("select * from raw.raw_line_items_map", con=engine,index_col="index")
    melt_actual_desc = pd.merge(data, pl_is_model_map,left_on="variable", right_on="variable",how="left",copy=False)
    return melt_actual_desc


def find_accounting_item_code(data):
    accounting_mapping = get_db_table_to_df("raw", "accounting item mapping")
    melt_actual_desc_code = pd.merge(data,
                                     accounting_mapping[['accounting_item_code', 'accounting_item_desc']],
                                     left_on='accounting_item_desc', right_on='accounting_item_desc', how="left")
    return melt_actual_desc_code


def most_recent_one(df):
    return df.sort_values(by="Version")[-1:]


def load_raw_accounting_to_melt_and_add_additional_cols(type="Budget"):
    type_filename = {"Budget": "budget p&l", "R3M":"r3m p&l", "Actual": "actual p&l"}
    file_db_name = type_filename[type]
    actual_pl = get_db_table_to_df("raw",file_db_name)
    actual_pl["Version"] = pd.to_datetime(actual_pl["Version"],format="%Y%m%d")
    actual_pl = actual_pl.groupby(["Period","Project"]).apply(most_recent_one)
    # TODO(如果11.1是null，用11.9填充，这块以后有数了可以删掉了)
    actual_pl["11.1 F/G+Zbox+base unit"] = actual_pl.apply(lambda x: x["11.1 F/G+Zbox+base unit"] if not
                                          np.isnan(x["11.1 F/G+Zbox+base unit"]) else x['11.9 Invoice quantity'],axis=1)
    melt_actual = actual_pl.melt(['Year', 'Month', 'Period', 'Site', 'Project','Version'])
    melt_actual_desc = find_accounting_item_desc(melt_actual)
    melt_actual_desc = melt_actual_desc[melt_actual_desc["accounting_item_desc"].notnull()]
    melt_actual_desc_code = find_accounting_item_code(melt_actual_desc)
    melt_actual_desc_code["currency"] = melt_actual_desc_code["accounting_item_desc"].apply(
        lambda x: None if re.search("quantity", x) else "NTD")
    melt_actual_desc_code["Period"] = pd.to_datetime(melt_actual_desc_code["Period"], format="%Y/%m")
    melt_actual_desc_code = find_fx_rate(melt_actual_desc_code, left_key="Period", type=type)
    melt_actual_desc_code["type"] = type
    melt_actual_desc_code["bg"] = "SDBG"
    melt_actual_desc_code["bu"] = ""
    return melt_actual_desc_code


def from_raw_to_modeltype(data,rename_dict):
    # budget and r3m to staging.tbl_is_modeltype
    data = data[data["accounting_item_group"].notnull() | data["currency"].isnull()]
    data.rename(columns=rename_dict,inplace=True)
    raw_to_stage_indireact("tbl_is_modeltype", data)


## r3m, budget to moh raw table
def actual_r3m_budget_to_moh_raw(data, type="R3M"):
    """
    :param data: actual buget r3m p&l raw data
    :param type: specify which one of those three
    :return: moh raw data; accounting_item_group是null
    """
    data = data[data["accounting_item_group"].isnull()]
    fixed_columns = ['Depreciation-W-buy', 'IDL', 'Facilities', 'other utilities']
    data["MOH_item_type"] = np.where(data["accounting_item_desc"].isin(fixed_columns),"Fixed", "Variable")
    data = find_model_code_in_project_info(data,left_key="Project")
    data["period_start"] = data["Period"]
    data["last_update_time"] = data["Period"]
    data["stat_dt"] = data["Period"]
    data["update_frequency"] = "monthly"
    moh_rename = {"AllieCode Name": "model_code",
                  "Project": "model_name",
                  "value": "amount",
                  }
    data.rename(columns=moh_rename, inplace=True)
    data["type"] = type

    moh_cols = ["Site","model_code","model_name","type","MOH_item_type","accounting_item_desc","amount",	"currency",
     "period_start","update_frequency",	"last_update_time",	"stat_dt"]
    moh_data = data[moh_cols]
    moh_data.to_sql("moh_raw",con=engine,schema="raw",if_exists="append")


def split_quantity_and_moh(moh_raw,amt_key="amount"):
    moh_raw = moh_raw[moh_raw["accounting_item_desc"] != "Invoice quantity"]
    moh_raw[amt_key] = moh_raw[amt_key].apply(lambda x: x if x>0 else np.nan)
    quantity = moh_raw[moh_raw["accounting_item_desc"] == "Production quantity"]
    total_moh = moh_raw[moh_raw["accounting_item_desc"] != "Production quantity"]
    return quantity,total_moh


def raw_to_staging_tbl_kpi_model():
    """
    :return: from raw to staging.tbl_kpi_model
    """
    ads = get_db_table_to_df("raw", "ADS")   # "Model"
    apr = get_db_table_to_df("raw", "apr")   # "project",
    mps = get_db_table_to_df("raw", "MPS")   # project
    scrap = get_db_table_to_df("raw", "Scrap amount") # project    in USD
    yield_rate = get_db_table_to_df("raw", "yield rate") # Model
    max_capacity = get_db_table_to_df("raw", "Max capacity") # Model
    apr.rename(columns={"project":"Model"}, inplace=True)
    mps.rename(columns={"project":"Model"}, inplace=True)
    scrap.rename(columns={"W_Model":"Model"}, inplace=True)

    scrap["Month"] = scrap["Month"].dt.strftime("%Y-%m-01")              # scrap月末数据变月初
    scrap["Month"] = scrap["Month"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"))
    data = pd.merge(ads[['Model', 'Month', 'ADS', 'COGS NTD','Inventory NTD', 'ads_target']],
                    apr[["Month", "Model", "APR _volume"]], left_on=["Model", "Month"], right_on=["Model", "Month"],
                    how="outer",copy=False)
    data = pd.merge(data, mps[["Month", "Model", "MPS _volume"]],left_on=["Model", "Month"], right_on=["Model", "Month"],how="outer",copy=False)
    data = pd.merge(data, scrap[['Model', 'Month', 'Actual Scrap AMT_USD',"scrap_rate"]],left_on=["Model", "Month"], right_on=["Model", "Month"],how="outer",copy=False)
    # 2019.7.9 add yield_rate_final_target in yield rate raw data
    data = pd.merge(data, yield_rate[['Month', 'Model', 'Yield rate','yield_rate_current_target','yield_rate_final_target']],
                    left_on=["Model", "Month"], right_on=["Model", "Month"],how="outer",copy=False)
    data = pd.merge(data, max_capacity[['Month', 'Model', 'Units']],left_on=["Model", "Month"], right_on=["Model", "Month"],how="outer",copy=False)
    data = find_fx_rate(data, left_key="Month")                                       # find usd to NTD fx rate
    data["Actual Scrap AMT_USD"] = data["Actual Scrap AMT_USD"] * data["FX"]          # change usd to NTD
    data["currency"] = "NTD"
    data["bg"] = "SDBG"
    data["bu"] = ""
    # find model site by model name
    data = find_site_in_project_info(data, left_key="Model")

    # rename raw data to map db col name
    raw_to_kpi_rename={
        'Site':"site",
        'Model': "product",
        'Month': "reporting_month",
        'ADS': "ads",
        'COGS NTD': "cogs",
        'Inventory NTD':"inventory",
        'Actual Scrap AMT_USD': "fg_scrap_amount",
        'Yield rate':"yield_rate",
        'APR _volume': "apr",
        'MPS _volume': "mps",
        'Units': "max_capacity_volume",
        'FX': "fx_rate"
    }
    data2 = data.rename(columns=raw_to_kpi_rename)
    raw_to_stage_indireact("tbl_kpi_model", data2)


def raw_to_dim_project():
    # project info(raw) to dim project(staging.dim_project)
    project_info = get_db_table_to_df("raw", "project info")
    project_info_rename = {
        'Project name': "product",
        'Site':"site",
        'AllieCode Name': "product_code",
        'Product Type': "product_type",
        'Current stage': "project_status",
        'MP Start': "mp_start_month",
        'NPI Start': 'npi_start_month',
        'Shipment Start': "shipment_start_month"
    }
    project_info["bg"] = "SDBG"
    project_info["bu"] = ""
    project_info.rename(columns=project_info_rename,inplace=True)
    raw_to_stage_indireact("dim_project", project_info)


if __name__ == '__main__':

    """
        project info to staging
    """
    raw_to_dim_project()


    """
        fx rate to staging
    """
    direct_raw_to_stage(raw_tb_name="FX rate", db_tb_name="fx_rate")


    """
        from raw to staging.tbl_kpi_model
    """
    raw_to_staging_tbl_kpi_model()


    """
            Actual, budget and R3M to staging.tbl_is_modeltype
            Actual first goes to is_model(db), then from is_model to staging.tbl_is_modeltype
            budget and r3m from excel raw table to staging.tbl_is_modeltype, the only difference is R3M need to pick the 
            latest version date given "Period"
            
            when accounting_item_group=null, Actual, budget and R3M go to raw.moh_raw, otherwise, they go to staging.tbl_is_modeltype
    """

    ## Actual
    #Step1: Actual P&L to is_model(in db)
    # TODO(this process is claimed to be one-time only, actual p&l data is loaded from is_model thereafter,this step should be deleted then)
    excel_file_name = "Actual p&l for loading history only.xlsx"
    actual_pl = pd.read_excel(os.path.join(raw_file,excel_file_name))
    # TODO(如果11.1是null，用11.9填充，这块以后有数了可以删掉了)
    actual_pl["11.1 F/G+Zbox+base unit"] = actual_pl.apply(lambda x: x["11.1 F/G+Zbox+base unit"] if not
                                          np.isnan(x["11.1 F/G+Zbox+base unit"]) else x['11.9 Invoice quantity'],axis=1)
    melt_actual = actual_pl.melt(['Year', 'Month', 'Period', 'Site', 'Project'])
    melt_actual_desc = find_accounting_item_desc(melt_actual)
    melt_actual_desc = melt_actual_desc[melt_actual_desc["accounting_item_desc"].notnull()]
    melt_actual_desc_code = find_accounting_item_code(melt_actual_desc)
    melt_actual_desc_code["currency"] = melt_actual_desc_code["accounting_item_desc"].apply(lambda x: None if re.search("quantity",x) else "NTD")
    melt_actual_desc_code["Period"] = pd.to_datetime(melt_actual_desc_code["Period"],format="%Y/%m")
    melt_actual_desc_code["last_update_time"] = melt_actual_desc_code["Period"]
    melt_actual_desc_code["stat_dt"] = melt_actual_desc_code["Period"]
    melt_actual_desc_code["update_frequency"] = "monthly"
    melt_actual_desc_code["model_code"] = ""
    melt_actual_desc_code["profit_center"] = "PCA000"
    melt_actual_desc_code["profit_center_name"] = "SDBG"

    actual_is_model_rename = {
    'Period': "period_start",
    'Project': "model_name",
    'value': "amount"
    }
    melt_actual_desc_code.rename(columns=actual_is_model_rename,inplace=True)
    is_model_cols = ["profit_center", "profit_center_name",	"model_code","model_name",	"accounting_item_code",	"accounting_item_group","accounting_item_desc",
                     "amount", "currency",	"period_start",	"last_update_time",	"stat_dt",	"update_frequency"]
    final_actual_to_is_model = melt_actual_desc_code[is_model_cols]
    is_model = final_actual_to_is_model[final_actual_to_is_model["accounting_item_group"].notnull()|final_actual_to_is_model["currency"].isnull()]
    is_model.to_sql("is_model",con=engine,schema="raw")


    #Step2: actual from is_model to staging.tbl_is_modeltype
    is_model_actual = get_db_table_to_df("raw", "is_model")
    is_model_actual["type"] = "Actual"
    is_model_actual = find_site_in_project_info(is_model_actual, left_key="model_name")
    is_model_actual = find_fx_rate(is_model_actual, left_key="period_start")
    is_model_actual["bu"] = ""
    is_model_actual["version"] = is_model_actual["period_start"].apply(lambda x: datetime(x.year-1,12,1))
    is_model_to_stage_rename = {
        'period_start': 'reporting_month',
        'profit_center_name': "bg",
        'Site': "site",
        'model_name': "product",
        'accounting_item_group': "line_itm_group",
        'accounting_item_desc': "line_item",
        'amount': "value",
        'FX': "fx_rate"
    }
    is_model_actual.rename(columns=is_model_to_stage_rename,inplace=True)
    raw_to_stage_indireact("tbl_is_modeltype", is_model_actual)


    # budget to staging.tbl_is_modeltype and raw.moh_raw
    budget_r3m_rename = {'Period': "reporting_month",
                     'Site': 'site',
                     'Project': "product",
                     'accounting_item_group': "line_itm_group",
                     'accounting_item_desc': "line_item",
                     'amount': "value",
                     'FX': "fx_rate",
                     'Version': "version"
                     }
    budget = load_raw_accounting_to_melt_and_add_additional_cols(type="Budget")
    from_raw_to_modeltype(budget,rename_dict=budget_r3m_rename)             # Budget to staging.tbl_is_modeltype
    actual_r3m_budget_to_moh_raw(budget, type="Budget")                     # Budget to raw.moh_raw

    # r3m to staging.tbl_is_modeltype and raw.moh_raw
    r3m = load_raw_accounting_to_melt_and_add_additional_cols(type="R3M")
    from_raw_to_modeltype(r3m,rename_dict=budget_r3m_rename)                   # r3m to staging.tbl_is_modeltype
    actual_r3m_budget_to_moh_raw(r3m, type="R3M")                                     # r3m to raw.moh_raw

    # actual to raw.moh_raw
    actual = load_raw_accounting_to_melt_and_add_additional_cols(type="Actual")
    actual_r3m_budget_to_moh_raw(actual, type="Actual")                                     # actual to raw.moh_raw


    """
        from raw.moh_raw to staging.tbl_unitmoh_monthly
    """

    # accouting item 里去掉invoice quantity, 把production quantity拿出来作为列，其余的item为cost，用cost除以quantity作为unit cost
    moh_raw = pd.read_sql("select * from raw.moh_raw", con=engine,index_col="index")
    moh_raw = moh_raw[['Site', 'model_name', 'type', 'accounting_item_desc', 'amount', 'period_start']]
    quantity, total_moh = split_quantity_and_moh(moh_raw)
    p_quantity = quantity.pivot_table(index=['period_start','Site', 'model_name', 'accounting_item_desc', ],
                                columns="type", values="amount").reset_index()
    p_moh = total_moh.pivot_table(index=['period_start','Site', 'model_name', 'accounting_item_desc', ],
                                columns="type", values="amount").reset_index()
    p_quantity.columns = ['period_start', 'Site', 'model_name', 'accounting_item_desc',
                          'Actual_quantity','Budget_quantity', 'R3M_quantity']
    unit_moh = pd.merge(p_moh, p_quantity[['period_start','model_name','Actual_quantity','Budget_quantity', 'R3M_quantity']],
                        how="left", left_on=['period_start', 'model_name'], right_on=['period_start', 'model_name'])
    unit_moh["Actual"] = unit_moh.eval("Actual / Actual_quantity")
    unit_moh["Budget"] = unit_moh.eval("Budget / Budget_quantity")
    unit_moh["R3M"] = unit_moh.eval("R3M / R3M_quantity")
    unit_moh.drop(['Actual_quantity','Budget_quantity', 'R3M_quantity'],axis=1, inplace=True)


    # add unit_moh_quote from below(2019.7.1 change as requested)
    # quote is in usd, need to change to NTD later
    quote = pd.read_sql("select * from raw.[quote p&l]", con=engine, index_col="index")
    quote = quote.melt(['Year', 'Month', 'Period', 'Site', 'Project'])
    quote = find_accounting_item_desc(quote)
    quote = quote[quote["accounting_item_desc"].notnull()]
    quote = find_fx_rate(quote,left_key="Period")
    quote_quantity, quote_moh = split_quantity_and_moh(quote,amt_key="value")
    quote_quantity.rename(columns={"value":"quantity"}, inplace=True)
    quote_unitmoh = pd.merge(quote_moh, quote_quantity[["Period","Project","quantity"]],how="left",
                             left_on=["Period","Project"], right_on=["Period","Project"])
    # moh quote from raw quote p&l is in USD
    quote_unitmoh["value"] = quote_unitmoh.eval("value * FX")              # 本来就是unit的了，不需要再除以quantity
    merged = pd.merge(unit_moh, quote_unitmoh[["Period", "Project", "accounting_item_desc", "value"]],
             left_on=["period_start", "model_name", "accounting_item_desc"],
             right_on=["Period", "Project", "accounting_item_desc"], how="left", copy=False)
    p_moh_rename={
    "period_start": "reporting_month",
    "Site": "site",
    "model_name": "product",
    "accounting_item_desc": "moh_item",
    "Actual": "unit_moh_actual",
    "Budget": "unit_moh_budget",
    "R3M": "unit_moh_r3m",
    "value": "unit_moh_quote"
    }
    merged.rename(columns=p_moh_rename, inplace=True)
    merged["bg"] = "SDBG"
    merged["bu"] = ""
    merged["currency"] = "NTD"
    raw_to_stage_indireact("tbl_unitmoh_monthly", merged)









