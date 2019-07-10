"""
@time: 7/3/2019 10:53 AM

@author: 柚子
"""

import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from raw_to_staging import aligh_table_cols_with_db,get_db_table_to_df,find_fx_rate

"""
    from staging to app
"""

engine = create_engine('mssql+pymssql://localhost:1433/master',echo=True)

def get_latest_date_from_is_model():
    is_model = get_table_from_staging("tbl_is_modeltype")
    filter_actual = is_model.query("type=='Actual'")
    date_time = pd.to_datetime(filter_actual["reporting_month"]).sort_values()
    latest_date = date_time[-1:].iloc[0]
    return latest_date


def replace_inf(data):
    data = data.replace(np.inf,np.nan)
    data = data.replace(-np.inf, np.nan)
    return data


def get_table_from_staging(table_name,index="id"):
    data = pd.read_sql(f"select * from staging.[{table_name}]", con=engine, index_col=index)
    return data


def get_table_from_app(table_name, index="id"):
    data = pd.read_sql(f"select * from app.[{table_name}]", con=engine, index_col=index)
    return data


def stage_to_app_indireact(db_tb_name, data2, db_id=True):
    """
    :param db_tb_name: app table that data2 goes to
    :param data2:
    :return: use app table cols to aquire data2's data and insert into app table
    """
    tbl_kpi = pd.read_sql(f"select * from app.[{db_tb_name}]",con=engine)
    db_data = aligh_table_cols_with_db(tbl_kpi,data2, remove_id=db_id)
    db_data.to_sql(db_tb_name, con=engine, schema="app",if_exists="append",index=False)


def get_line_item_merge_map_dict():
    line_item_merge_map = pd.read_sql("select * from raw.line_item_merge_map", con=engine,
                                      index_col="accounting_item_desc")
    mapping = line_item_merge_map["app_item"].to_dict()
    return mapping


def find_variances_in_line_items(x):
    if x == 'Material cost_Purchase price variance':
        cat = "price_variance"
    elif x == 'Material cost_Material usage variance':
        cat = "usage_variance"
    elif x == "Sales_Normal shipment":
        cat = "shipment"
    elif x == "Sales_Sales return & allowance":
        cat = "allowance"
    else:
        cat = None
    return cat


def get_project_status(data, left_key="product"):
    # find project status(aim for MP) in staging.dim_project
    project_info = get_db_table_to_df("staging", "dim_project")
    data = pd.merge(data, project_info[['product', "project_status"]], left_on=left_key,
                    right_on=["product"], how="left", copy=False)
    return data

def get_product_type(data, left_key="product"):
    # look for product type, i.e. Phone, accessory
    project_info = get_db_table_to_df("staging", "dim_project")
    data = pd.merge(data, project_info[['product', 'product_type']], left_on=left_key,
                    right_on=["product"], how="left", copy=False)
    return data

def find_cost_cat_code(data, left_key="cost_item_level1"):
    # look for cost category code by cost category name
    cost_cat = get_db_table_to_df("app", "cost_category")
    data = pd.merge(data, cost_cat[['cost_cat_code', 'cost_cat_name']], left_on=left_key,
                    right_on='cost_cat_name',how="left")
    return data


def find_cost_code_and_non_adj(data, left_key="cost_item_level1"):
    # look for cost non_adj_flag by cost category name
    cost_cat = get_db_table_to_df("app", "cost_category")
    data = pd.merge(data, cost_cat[['cost_cat_code', 'non_adj_flag', 'cost_cat_name']], left_on=left_key,
                    right_on='cost_cat_name',how="left", copy=False)
    return data


def shipment_apr(x):
    if x["apr"] > 0:
        return min(x["shipment_volume"] / (x["apr"]), 1) if x["apr"] > 0 else np.nan


def production_apr(x):
    return min(x['production_volume'] / (x["apr"]), 1) if x["apr"] > 0 else np.nan


def moh_quote_score(x):
    return max(0,min(1,1-(x["unit_moh_actual"]-x["unit_moh_quote"])/(x["unit_moh_quote"]))) if x["unit_moh_actual"] > 0 else np.nan


def ads_vs_target(x):
    return max(0,min(1,1-(x["ads"]-x['ads_target']) / (x['ads_target']))) if x['ads'] > 0 else np.nan


def yield_vs_target(x):
    return min(x['yield_rate'] / (x["yield_rate_current_target"]), 1) if x["yield_rate_current_target"] > 0 else np.nan


def material_var_vs_0target(x):
    return max(min(1,1-x["material_var"]*100),0) if not np.isnan(x["material_var"]) else np.nan


def cal_six_dimension_score(raw_matrix):
    """
    :param raw_matrix: inital input for health score calculation for each model and each month
    :return: six dimension score for each model and each month
    """
    six_dimension_matrix = raw_matrix[["reporting_month", "product"]]
    six_dimension_matrix["shipment/apr"] = raw_matrix.apply(shipment_apr, axis=1)
    six_dimension_matrix["production/apr"] = raw_matrix.apply(production_apr, axis=1)
    six_dimension_matrix["moh/quote"] = raw_matrix.apply(moh_quote_score, axis=1)
    six_dimension_matrix["ads/ads_target"] = raw_matrix.apply(ads_vs_target, axis=1)
    six_dimension_matrix["yield/yield_target"] = raw_matrix.apply(yield_vs_target, axis=1)
    six_dimension_matrix["material/m_target"] = raw_matrix.apply(material_var_vs_0target, axis=1)
    return six_dimension_matrix

def load_six_dimension_weights():
    # for health score counting
    weights = pd.read_sql("select * from raw.health_score_weights", con=engine, index_col="idx")
    weights = pd.Series(weights["weight"])
    return weights


def cal_weight_average_health_score(six_dimension_matrix, weights):
    index = six_dimension_matrix[["reporting_month", "product"]]
    matrix = six_dimension_matrix.drop(["reporting_month", "product"], 1)
    health_score_matrix = matrix.mul(weights, 1).sum(1) / matrix.notnull().mul(weights, 1).sum(1)
    health_score_df = pd.Series(health_score_matrix, name="health_score")
    health_score = pd.concat([index, health_score_df], 1)
    return health_score


def cal_health_score(kpi_rev_moh):
    product_status = get_project_status(kpi_rev_moh)  # find product status of products
    mp_products = product_status.query("project_status=='MP'")    # only mp status projects will count health score
    mp_products["material_var"] = mp_products.apply(
        lambda x: x["material_total_variance"] / x["revenue"] if x["revenue"] > 0 else np.nan, axis=1)

    # step1: select columns that are involved in health score calculation, raw_matrix is the first step
    raw_matrix = mp_products[["reporting_month", "product", "revenue", 'shipment_volume', "apr", 'production_volume',
                              'unit_moh_actual', 'unit_moh_quote', "ads", 'ads_target', 'yield_rate',
                              'yield_rate_current_target', "material_var"]]. \
        sort_values(by=["product", "reporting_month"])

    six_dimension_matrix = cal_six_dimension_score(raw_matrix)  # step2: calculate six dimension score,using raw_matrix
    weights = load_six_dimension_weights()    # step3: load weights for calculating 6 dimensions weighted average score
    health_score = cal_weight_average_health_score(six_dimension_matrix, weights)  # step4: calculating weighted-average score of 6 dimension for each model and each month
    return health_score


def cal_inventory_2months_avg(df):
    df = df.sort_index(by="reporting_month")
    df["inventory"] = df["inventory"].rolling(2).mean()
    return df


# def cal_actual_bg_ads_and_scrap_rate(df):
#     ads = df["avg_inventory_for_ads"].sum() / df["cogs_for_ads"].sum() * 30
#     scrap = df["fg_scrap_amount"].sum() / df["revenue_amt_for_scrap"].sum()
#     return pd.Series({"ads":ads, "scrap_rate":scrap})


def bg_target_ads_scrap_rate(df):
    ads = df["fg_ads_target"].values[0]
    scrap = df["fg_scrap_rate_target"].values[0]
    return pd.Series({"ads":ads, "scrap_rate":scrap})


def filter_reporting_month_by_year(data, date_key="reporting_month", from_year_on=2018):
    # filter app data's date,only keeps data from certain year on
    if data[date_key].dtype == 'O':
        data[date_key] = pd.to_datetime(data[date_key])
    return data[data[date_key].dt.year >= from_year_on]


def filter_reporting_month_by_date(data, selected_date, date_key="reporting_month"):
    if data[date_key].dtype == 'O':
        data[date_key] = pd.to_datetime(data[date_key])
    return data[data[date_key] == selected_date]


def get_table_from_dim(table_name,idx="id"):
    return pd.read_sql(f"select * from dim.[{table_name}]", con=engine, index_col=idx)


def total_cost_for_cost_overview(source_item):
    """
    :param source_item:  each cost item level1 and each cost item level2 for each month each product
    :return: total cost for actual, r3m and budget
    """
    stage_unitmoh_monthly = get_table_from_staging("tbl_unitmoh_monthly")
    stage_modeltype = get_table_from_staging("tbl_is_modeltype")
    # partition into two parts, calculate seperately
    source_for_moh = source_item.query("(cost_item_level1 == 'MOH_WEKS') | (cost_item_level1 == 'MOH_WMI')")
    source_for_modeltype = source_item.query("~((cost_item_level1 == 'MOH_WEKS') | (cost_item_level1 == 'MOH_WMI'))")

    stage_unitmoh_monthly["cost_item_level2"] = stage_unitmoh_monthly.loc[:, "moh_item"]
    source_moh = pd.merge(source_for_moh,stage_unitmoh_monthly[['reporting_month', 'product','cost_item_level2',
                                                                'unit_moh_actual', 'unit_moh_budget','unit_moh_r3m']],
                          left_on=["reporting_month", "product", "cost_item_level2"],
                          right_on=["reporting_month", "product", "cost_item_level2"],
                          how="left")
    source_moh.rename(columns={'unit_moh_actual': "actual_total_cost",
                               'unit_moh_r3m': 'r3m_total_cost',
                               'unit_moh_budget': 'budget_total_cost'}, inplace=True)
    def merge_is_modeltype_use_type(type="Actual"):
        filtered_data = stage_modeltype.query("type==@type")
        filtered_data["cost_item_level1"] = filtered_data['line_item']
        merged = pd.merge(source_for_modeltype, filtered_data[['reporting_month', 'product',"cost_item_level1","value"]],
                 left_on=["reporting_month", "product", "cost_item_level1"],
                 right_on=['reporting_month', 'product',"cost_item_level1"],how="left")
        type = type.lower()
        merged.rename(columns={"value":f"{type}_total_cost"}, inplace=True)
        return merged

    for type in ["Actual", "Budget", "R3M"]:
        source_for_modeltype = merge_is_modeltype_use_type(type = type)
    source_item_with_total_cost = pd.concat([source_moh, source_for_modeltype])
    return source_item_with_total_cost


def shipment_for_cost_overview(source_item):
    """
    :param source_item:
    :return: find shipment and production for each model each month and for actual budget r3m as cols seperately
    """
    stage_modeltype = get_table_from_staging("tbl_is_modeltype")

    def query_is_modeltype(type="Actual"):
        filter_is_model = stage_modeltype.query("type==@type")
        ship_production = filter_is_model.query("(line_item=='Invoice quantity') | (line_item=='Production quantity')")
        unstacked = ship_production.pivot_table(index=['reporting_month','product'],columns="line_item",values="value")
        type = type.lower()
        unstacked.rename(columns={'Invoice quantity':f"{type}_shipment",
                                  'Production quantity': f"{type}_production"},inplace=True)
        return unstacked

    for type in ["Actual", "Budget", "R3M"]:
        segment = query_is_modeltype(type)
        source_item = pd.merge(source_item, segment,left_on=['reporting_month','product'],right_index=True,how="left")

    return source_item


def money_from_ntd_to_usd(data, cols, fx_col_name="FX"):
    for col in cols:
        data[col] = data.eval(f"{col}/{fx_col_name}")
    return data


def mul_cost_with_production(data):
    # 2019.7.5计算total cost的时候需要乘上production
    # 2019.7.7 仅仅(cost_item_level1 == 'MOH_WEKS') | (cost_item_level1 == 'MOH_WMI')的时候需要做这个乘法
    condition = (data["cost_item_level1"] == 'MOH_WEKS') | (data["cost_item_level1"] == 'MOH_WMI')
    for col in ["actual", "budget", "r3m"]:
        cost = f"{col}_total_cost"
        production = f"{col}_production"
        data[cost] = np.where(condition, data[cost] * data[production], data[cost])
    return data


def divide_total_cost_by_shipment_production(data, divided_by="shipment"):
    """
    :param data:
    :param divided_by:
    :return: for luke
    """
    cols = list(data.columns)
    cols.remove(divided_by)
    for col in cols:
        data[col] = data.apply(lambda x: x[col] / x[divided_by] if x[divided_by] > 0 else np.nan, axis=1)
    return data


def add_usd_currency_and_fx_rate_for_cost_app_tables(data):
    data["currency"] = "USD"
    data = find_fx_rate(data,left_key="reporting_month")
    data.rename(columns={"FX":"fx_rate"}, inplace=True)
    return data

def add_NTD_currency_and_fx_rate(data):
    data["currency"] = "NTD"
    data = find_fx_rate(data, left_key="reporting_month")
    data.rename(columns={"FX":"fx_rate"}, inplace=True)
    return data

"""
    from below are main etl functions
    i.e. each function create an app table
    1.app.perf_kpi_model

"""

def staging_tbl_kpi_model_to_app_perf_kpi_model():
    """
        from staging to app.perf_kpi_model(previous name app.tbl_kpi_model
    """
    # Step1. process staging.kpi_model
    # except inventory, only rename is needed, no calculation need, inventory need to calculate 2 days average(this month and last month
    stage_kpi = get_table_from_staging("tbl_kpi_model")
    stage_kpi = stage_kpi.groupby("product").apply(cal_inventory_2months_avg).reset_index(drop=True)
    stage_kpi_rename = {"cogs": "cogs_for_ads",
                        "inventory": "avg_inventory_for_ads",
                        "ads_target": "fg_ads_target",
                        }
    stage_kpi.rename(stage_kpi_rename, inplace=True)

    # Step2. process staging.is_modeltype
    stage_modeltype = get_table_from_staging("tbl_is_modeltype")
    mapping = get_line_item_merge_map_dict()
    app_map = stage_modeltype['line_item'].map(mapping)
    summary = stage_modeltype.groupby(['reporting_month', 'product', 'type', app_map])['value'].sum().unstack()
    summary['mva'] = summary.eval('revenue - material_cost')
    summary['operating_income'] = summary.eval('mva - other_cost - opex - moh')
    variance_map = stage_modeltype['line_item'].apply(find_variances_in_line_items)
    summary2 = stage_modeltype.groupby(['reporting_month', 'product', 'type', variance_map])['value'].sum().unstack()
    summary2["material_total_variance"] = summary2.eval("price_variance + usage_variance")
    summary2["revenue_amt_for_scrap"] = summary2.eval("shipment + allowance")
    combine_summary = pd.merge(summary, summary2[["material_total_variance", "revenue_amt_for_scrap"]],
                               left_index=True, right_index=True, how="left", copy=False)
    combine_summary["material_unit_variance"] = combine_summary.apply(
        lambda x: x["material_total_variance"] / x["production_volume"] if x["production_volume"] > 0 else np.nan,
        axis=1)

    actual_stats = combine_summary.query("type=='Actual'").reset_index()
    kpi_rev = pd.merge(stage_kpi, actual_stats,
                       left_on=["reporting_month", "product"], right_on=["reporting_month", "product"], how="left",
                       copy=False)

    #Step3. process staging.tbl_unitmoh_monthly
    # add 'unit_moh_actual', 'unit_moh_quote' to current data kpi_rev
    unitmoh = get_table_from_staging("tbl_unitmoh_monthly")
    actual_moh = unitmoh.groupby(["reporting_month", "product"])[
        'unit_moh_actual', 'unit_moh_quote'].sum().reset_index()
    kpi_rev_moh = pd.merge(kpi_rev, actual_moh, left_on=["reporting_month", "product"],
                           right_on=["reporting_month", "product"], how="left", copy=False)

    #Step4. calculate health score
    health_score = cal_health_score(kpi_rev_moh)

    # add health score to app.perf_kpi_model
    app_kpi = pd.merge(kpi_rev_moh, health_score, left_on=["reporting_month", "product"],
                       right_on=["reporting_month", "product"], how="left", copy=False)
    app_kpi_rename = {"ads_target": "fg_ads_target",
                      "inventory": "avg_inventory_for_ads",
                      'cogs': 'cogs_for_ads',
                      'scrap_rate': "fg_scrap_rate_target",
                      "yield_rate_current_target":"yield_rate_curr_target",
                      "max_capacity_volume":"max_capacity",
                      "production_volume": "production",
                      "shipment_volume": "shipment"
                      }
    app_kpi.rename(columns=app_kpi_rename, inplace=True)
    filter_year_app_kpi = filter_reporting_month_by_year(app_kpi)
    stage_to_app_indireact("perf_kpi_model", filter_year_app_kpi)


def staging_is_kpi_moh_to_app_perf_overview_model():
    ## part1 staging.tbl_is_modeltype
    stage_modeltype = get_table_from_staging("tbl_is_modeltype")
    mapping = get_line_item_merge_map_dict()
    stage_modeltype["cat"] = stage_modeltype['line_item'].map(mapping)
    summary = stage_modeltype.groupby(['reporting_month', 'bg', 'site', 'product', 'type', "cat"])[
        'value'].sum().unstack()
    summary['mva'] = summary.eval('revenue - material_cost')
    summary['operating_income'] = summary.eval('mva - other_cost - moh - opex')

    # 2019.7.10 luke asked to add ads and scrap rate
    # get revenue_amt_for_scrap
    variance_map = stage_modeltype['line_item'].apply(find_variances_in_line_items)
    summary2 = stage_modeltype.groupby(['reporting_month', 'product', 'type', variance_map])[
        'value'].sum().unstack()
    summary2["revenue_amt_for_scrap"] = summary2.eval("shipment + allowance")
    combine_summary = pd.merge(summary, summary2[["revenue_amt_for_scrap"]],
                               left_index=True, right_index=True, how="left", copy=False)

    # 2019.7.10 luke asked to add ads and scrap rate
    ## part2 staging.tbl_kpi_model --- get fg_scrap_amt
    stage_kpi = get_table_from_staging("tbl_kpi_model")
    stage_kpi["type"] = "Actual"
    merge_index = ['reporting_month', 'bg', 'site', 'product', 'type']
    stage_kpi = stage_kpi.set_index(merge_index)
    kpi_is_summary = combine_summary.merge(stage_kpi[["ads", "fg_scrap_amount"]], on=merge_index, how="left")
    kpi_is_summary["scrap_rate"] = kpi_is_summary.eval("fg_scrap_amount / revenue_amt_for_scrap")

    ## part3 staging.unitmoh_monthly
    moh_quote = get_table_from_staging("tbl_unitmoh_monthly")
    moh_quote = moh_quote.query("moh_item!='Production quantity'")
    unit_moh_quote = pd.DataFrame(moh_quote.groupby(['reporting_month', 'product'])["unit_moh_quote"].sum())
    merged = pd.merge(kpi_is_summary.reset_index(), unit_moh_quote.reset_index(),
                      left_on=["reporting_month", "product"],
                      right_on=["reporting_month", "product"], how="left")
    merged["currency"] = "NTD"
    merged["bu"] = ""
    merged = find_fx_rate(merged, left_key="reporting_month")
    app_perf_overview_rename = {"production_volume": "production",
                                "shipment_volume": "shipment",
                                "FX": "fx_rate"}
    merged.rename(columns=app_perf_overview_rename, inplace=True)
    filter_year = filter_reporting_month_by_year(merged)
    stage_to_app_indireact("perf_overview_model", filter_year)


def get_cat_ind(level1_shipment, type):
    if type == "site_and_type":
        level1_shipment["new_cat_ind"] = level1_shipment["cat_ind"]
    elif type == "all":
        level1_shipment["new_cat_ind"] = "ALL"
    elif type == "site":
        level1_shipment["new_cat_ind"] = level1_shipment["cat_ind"].apply(lambda x: x[:3])
    else:  # product_type
        level1_shipment["new_cat_ind"] = level1_shipment["cat_ind"].apply(lambda x: x[4:])
    return level1_shipment


def perf_overview_all_by_cat_ind(level1_shipment, type="site_and_type"):
    # 计算site_product_type两级,工厂级，product_type级，还有所有汇总（all)

    index_cols = ['bg', 'reporting_month', 'new_cat_ind', 'type']
    value_cols = ['revenue', 'mva', 'material_cost', 'other_cost', 'opex', 'operating_income', 'moh',
                  'production', 'shipment','ads','scrap_rate']
    level1_shipment = get_cat_ind(level1_shipment, type)
    group_cols = index_cols
    app_kpi = level1_shipment.groupby(group_cols).sum()
    # health score rate calculations
    app_kpi["scrap_rate"] = app_kpi.eval("fg_scrap_amount / revenue_amt_for_scrap")
    app_kpi["ads"] = app_kpi.eval("avg_inventory_for_ads / cogs_for_ads * 30")
    cat_ind_group = app_kpi[value_cols]
    cat_ind_group = cat_ind_group.reset_index()
    cat_ind_group.rename(columns={"new_cat_ind": "cat_ind"}, inplace=True)
    return cat_ind_group


def app_perf_model_to_perf_overview_all():
    # columns= [reporting_month, bg, cat_ind, type, revenue, mva, operating_income, moh, production, shipment, ads,
    # scrap_rate,material_cost,other_cost opex]
    # for budget and r3m, ads and scrap rate is target rate
    # for actual, ads and scrap rate need calculation

    # step1: get data source1 by model for calculate actual budget r3m ads and scrap rate
    app_kpi = get_table_from_app("perf_kpi_model")          # type=actual
    target = {"fg_ads_target": app_kpi["fg_ads_target"].values[0],
              'fg_scrap_rate_target': app_kpi['fg_scrap_rate_target'].values[0]}
    kpi_cols = ['reporting_month', 'product',
                'cogs_for_ads','avg_inventory_for_ads', 'fg_scrap_amount', 'revenue_amt_for_scrap']
    app_kpi = app_kpi[kpi_cols]
    app_kpi["type"] = "Actual"
    # step2. get data source2 by model for income, cost, quantity
    app_perf = get_table_from_app("perf_overview_model")
    use_cols = ['reporting_month', 'bg', 'site', 'product', 'type',
                'revenue','mva', 'material_cost', 'other_cost', 'opex', 'operating_income', 'moh',
                'production', 'shipment']
    app_perf = app_perf[use_cols]
    # step3. merge two data sources
    base_data = app_perf.merge(app_kpi, on=["reporting_month", "product", "type"], how="left")
    base_data = get_product_type(base_data)
    base_data["cat_ind"] = base_data.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    # get aggregate data by cat_ind
    perf_overview_all_data = combine_three_cat_ind_level(base_data, standard=None, groupfunc=perf_overview_all_by_cat_ind)
    # for budget and r3m, ads and scrap rate is target rate
    perf_overview_all_data["ads"] = np.where(perf_overview_all_data["type"]!='Actual', target["fg_ads_target"], perf_overview_all_data["ads"])
    perf_overview_all_data["scrap_rate"] = np.where(perf_overview_all_data["type"]!='Actual',
                                                    target["fg_scrap_rate_target"], perf_overview_all_data["ads"])
    perf_overview_all_data = add_NTD_currency_and_fx_rate(perf_overview_all_data)
    stage_to_app_indireact("perf_overview_all", perf_overview_all_data)


########################################################################################
                     # luke cost app table functions tools
########################################################################################


def groupby_and_per_shipment(level1_shipment, type="site_and_type", divided_by=None):
    # 计算site_product_type两级,工厂级，product_type级，还有所有汇总（all)
    level1_shipment = get_cat_ind(level1_shipment, type)
    group_cols = ['bg', 'reporting_month', 'new_cat_ind', 'type']
    cost_level1_all = level1_shipment.groupby(group_cols).sum()
    if divided_by:      # 有可能不需要计算cost per shipment, cost per production
        cost_level1_all = divide_total_cost_by_shipment_production(cost_level1_all, divided_by=divided_by)
    cost_level1_group = cost_level1_all.reset_index()  # 最后一列的shipment or production是正常的
    cost_level1_group.rename(columns={"new_cat_ind": "cat_ind"}, inplace=True)
    return cost_level1_group


def combine_three_cat_ind_level(level1_shipment, standard, groupfunc=groupby_and_per_shipment):
    # 计算site_producttype两级,工厂级，product_type级，还有汇总级（all)的多列cost 总和
    # standard=None,shipment or production,
    # ----None的时候表示是total cost加总
    # ----shipment表示先求和total cost,再除以shipment quantity的和
    # ----production表示先求和total cost,再除以production quantity的和
    shipment_all = pd.DataFrame()
    for agg_level in ["site_and_type", "site", "all", "product_type"]:
        if standard:
            standard = f"{standard}"  # 如果standard是None，即total cost汇总，非total cost per shipment/production
            cost_level1_group = groupfunc(level1_shipment, type=agg_level, divided_by=standard)
        else:
            cost_level1_group = groupfunc(level1_shipment, type=agg_level)
        shipment_all = pd.concat([shipment_all, cost_level1_group])
    return shipment_all


def cost_overview_level1(cost, p_cost_level1, standard="shipment"):
    """
    :param cost: cost_overview
    :param p_cost_level1: cost_overview without shipment and production, cost level1 are presented as multiple columns
    :param standard:
    :return: 先把actual/budget/r3m的shipment或production拿出来，然后和一堆total cost列merge到一起，
    然后对cat_ind按几种切分方法(site,product_type,all,site&product_type，计算sum(cost)/sum(production)
    """
    cost_shipment = cost.rename(columns={f"actual_{standard}":"Actual",
                                         f"budget_{standard}":"Budget",
                                         f"r3m_{standard}": "R3M"})
    cost_shipment = cost_shipment[['reporting_month', 'bg','cat_ind','product',"Actual","Budget", "R3M"]]
    cost_shipment_melt = cost_shipment.melt(['reporting_month', 'bg','cat_ind','product'])
    cost_shipment_melt = cost_shipment_melt.drop_duplicates()# 原始的不同line item会对应同一个production，所以会重复
    cost_shipment_melt.rename(columns={"variable":"type", "value":standard}, inplace=True)

    level1_shipment = pd.merge(p_cost_level1, cost_shipment_melt,how="left",
                               left_on=['reporting_month', 'bg','cat_ind','product','type'],
                               right_on=['reporting_month', 'bg','cat_ind','product','type'],copy=False)

    # 计算site_product_type两级,工厂级，product_type级，还有所有汇总（all)
    shipment_all = combine_three_cat_ind_level(level1_shipment, standard=standard)
    shipment_all["standard"] = standard
    return shipment_all


def groupby_level2_with_unit_cost(level1_shipment, type="site_and_type",divided_by="both"):
    # 计算site_product_type两级,工厂级，product_type级，还有所有汇总（all)
    level1_shipment = get_cat_ind(level1_shipment, type)
    group_cols = ['bg', 'reporting_month','type','product','cost_cat_code','cost_item_level2', 'new_cat_ind']
    cost_level1_all = level1_shipment.groupby(group_cols).sum()
    if divided_by == "both":
        cost_level1_all["unit_cost_ship"] = cost_level1_all.apply(
            lambda x: x["total_cost"] / x["shipment"] if x["shipment"] > 0 else np.nan, axis=1)
        cost_level1_all["unit_cost_prod"] = cost_level1_all.apply(
            lambda x: x["total_cost"] / x["production"] if x["production"] > 0 else np.nan, axis=1)
    cost_level1_group = cost_level1_all.reset_index()  # 最后一列的shipment or production是正常的
    cost_level1_group.rename(columns={"new_cat_ind": "cat_ind"}, inplace=True)
    return cost_level1_group


def extract_cost_or_shipment_or_production(cost, item):
    """
    :param cost:
    :param item:total_cost/shipment/production
    :return:
    """
    cost_apart = cost.rename(columns={f"actual_{item}": "Actual",
                                      f"budget_{item}": "Budget",
                                      f"r3m_{item}": "R3M"})

    cost_apart = cost_apart[["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1',
                             'cost_item_level2', 'Actual', 'Budget', "R3M"]]
    type_as_col = cost_apart.melt(["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1','cost_item_level2'])
    type_as_col.rename(columns={"variable": "type", "value": f"{item}"}, inplace=True)
    return type_as_col

################################################################################################
                            # luke health tables function tools
################################################################################################

def health_rate_agg_by_cat_ind(level1_shipment, type="site_and_type"):
    # 计算site_product_type两级,工厂级，product_type级，还有所有汇总（all)
    index_cols = ['bg', 'reporting_month', 'new_cat_ind']
    value_cols = ['shipment_rate', 'production_rate', 'unit_moh_rate', 'material_cost_variance', 'yield_rate',
                  'yield_rate_curr_target', "ads"]
    level1_shipment = get_cat_ind(level1_shipment, type)
    group_cols = index_cols
    level1_shipment["yield_volume"] = level1_shipment.eval("yield_rate * production")
    level1_shipment["yield_target_volume"] = level1_shipment.eval("yield_rate_curr_target * production")
    app_kpi = level1_shipment.groupby(group_cols).sum()
    # unit_moh_quote should calculate mean when aggregate by cat_ind
    avg_unit_moh_quote = level1_shipment.groupby(group_cols)["unit_moh_quote"].mean()
    app_kpi.update(avg_unit_moh_quote)
    # health score rate calculations
    app_kpi["shipment_rate"] = app_kpi.eval("shipment / apr")
    app_kpi["production_rate"] = app_kpi.eval("production / apr")
    app_kpi["material_cost_variance"] = app_kpi.eval("material_total_variance / revenue")
    app_kpi["unit_moh_rate"] = app_kpi.eval("moh / production / unit_moh_rate")
    app_kpi["material_cost_variance"] = app_kpi.eval("material_total_variance / revenue")
    app_kpi["yield_rate"] = app_kpi.eval("yield_volume / production")
    app_kpi["yield_rate_curr_target"] = app_kpi.eval("yield_target_volume / production")
    app_kpi["ads"] = app_kpi.eval("avg_inventory_for_ads / cogs_for_ads * 30")
    cat_ind_group = app_kpi[value_cols]
    cat_ind_group = cat_ind_group.reset_index()
    cat_ind_group.rename(columns={"new_cat_ind": "cat_ind"}, inplace=True)
    return cat_ind_group


def app_kpi_model_to_health_kpi_model():
    """
        from app.perf_overview_model & app.perf_kpi_model to app.health_kpi_model(for luke)
        # source1: app.perf_overview_model:moh
        # source2: app.perf_kpi_model:others
    """
    # Step1. process app.perf_kpi_model
    app_kpi = get_table_from_app("perf_kpi_model")
    app_kpi["shipment_rate"] = app_kpi.eval("shipment / apr")
    app_kpi["production_rate"] = app_kpi.eval("production / apr")
    app_kpi["material_cost_variance"] = app_kpi.eval("material_total_variance / revenue")
    # Step2. process perf_overview_model to get unit_moh_rate
    app_overview = get_table_from_app("perf_overview_model")
    app_overview = app_overview.query("type == 'Actual'")
    app_overview["unit_moh_rate"] = app_overview.eval("moh/production/unit_moh_quote")
    health_kpi = pd.merge(app_kpi, app_overview[["reporting_month", "moh", "product","unit_moh_rate"]],
                          on=["reporting_month", "product"], how="left")
    health_kpi = replace_inf(health_kpi)
    return health_kpi


def heath_kpi_model_to_health_kpi_group():
    raw_health_kpi_model = app_kpi_model_to_health_kpi_model()    # source data comes from app.health_kpi_model's source
    raw_health_kpi_model = get_product_type(raw_health_kpi_model)
    raw_health_kpi_model["cat_ind"] = raw_health_kpi_model.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    health_kpi_group = combine_three_cat_ind_level(raw_health_kpi_model, standard=None, groupfunc=health_rate_agg_by_cat_ind)
    health_kpi_group = replace_inf(health_kpi_group)
    stage_to_app_indireact("health_kpi_group", health_kpi_group)


if __name__ == '__main__':

    ########################################################################
                                # perf ETL #
    ########################################################################

    """
        from staging to app.perf_overview_model 
    """
    # app.perf_overview_model(previous name app.tbl_is_modeltype
    staging_is_kpi_moh_to_app_perf_overview_model()

    """
        from staging to app.perf_kpi_model(previous name app.tbl_kpi_model
    """
    # 2019.7.10 luke asked for app.health_kpi_model, many same cols in app.perf_kpi_model, for now both tables exist in db
    staging_tbl_kpi_model_to_app_perf_kpi_model()

    """
        app.perf_overview_all
        source data1: app.perf_kpi_model   usage: calculate monthly bg level ads and scrap rate
        source data2: perf_overview_model  usage: calculate monthly bg level performance_kpi's sum
    """
    app_perf_model_to_perf_overview_all()


    ################################################################################################
                                            # Health ETL #
    ################################################################################################

    # copy staging.dim_project to app.health_project_status
    dim_project = get_table_from_staging("dim_project")
    dim_project.to_sql("health_project_status", con=engine, schema="app")

    """
        from app to app.health_kpi_model
    """
    # 2019.7.10 luke asked for app.health_kpi_model, many same cols in app.perf_kpi_model, for now both tables exist in db
    health_kpi = app_kpi_model_to_health_kpi_model()
    stage_to_app_indireact("health_kpi_model", health_kpi)

    """
        from app.health_kpi_model source to app.health_kpi_group
    """
    # [id,reporting_month, bg, cat_ind,shipment_rate, production_rate, unit_moh_rate, material_cost_variance,yield_rate,
    # yield_rate_curr_target, ads]
    heath_kpi_model_to_health_kpi_group()

    #############################################################################################
                            # COST (Luke) ETL
    #############################################################################################

    """
        app.cost_overview_dtl
        source: staging.tbl_is_modeltype
                dim.tbl_cost_control_list
                staging.tbl_unitmoh_monthly 
    """
    source = get_table_from_staging("tbl_is_modeltype")
    trim_source = source[["reporting_month", "bg", "bu", "site","product"]].drop_duplicates()
    dim_cost = get_table_from_dim("tbl_cost_control_list")
    source_item = pd.merge(trim_source, dim_cost, how="outer")      # make each month each product has same full line items
    source_item.rename(columns={"detailed_item":"cost_item_level2",
                        "line_item":"cost_item_level1",
                        "line_itm_group": "cost_box_group"}, inplace=True)
    source_item_with_cost = total_cost_for_cost_overview(source_item) # add actual_total_cost, budget_total_cost, r3m_total_cost
    cost_overview = shipment_for_cost_overview(source_item_with_cost) # add actual/r3m/budget shipment and production
    cost_overview = find_fx_rate(cost_overview,left_key="reporting_month")
    # change money cols from ntd to usd
    cost_overview = money_from_ntd_to_usd(cost_overview, cols=['actual_total_cost', 'budget_total_cost', 'r3m_total_cost'])
    # 2019.7.5计算total cost的时候需要乘上production 这里补上
    # 2019.7.7 仅仅(cost_item_level1 == 'MOH_WEKS') | (cost_item_level1 == 'MOH_WMI')的时候需要做这个乘法
    cost_overview = mul_cost_with_production(cost_overview)
    # 2019.7.5site=WMI的时候cost_item_level1=MOH_WEKS的三个total cost改成0；site=WKS的时候，cost_item_level1=MOH_WMI的三个cost改成0
    cost_overview.loc[cost_overview.query("((site=='WMI')&(cost_item_level1=='MOH_WEKS'))|((site=='WKS')&(cost_item_level1=='MOH_WMI'))").index,\
    ["actual_total_cost", "budget_total_cost", "r3m_total_cost"]] = np.nan
    cost_overview["currency"] = "USD"
    cost_overview.rename(columns={"FX":"fx_rate"}, inplace=True)
    # filter 2018 onwards data
    filtered_cost_overview = filter_reporting_month_by_year(cost_overview, date_key="reporting_month", from_year_on=2018)

    stage_to_app_indireact("cost_overview_dtl",filtered_cost_overview)

    """
        2019.7.5 Luke's cost app tables
        1.app.cost_overview_level1
            
    """
    cost = get_table_from_app("cost_overview_dtl")
    cost = get_product_type(cost)
    cost["cat_ind"] = cost.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    cost = find_cost_cat_code(cost,left_key='cost_item_level1')
    # 计算site&product_type level的per shipment,per production
    # Step1. 拿cost出来,pivot成多列， actual,budget,r3m total cost从原来的三列melt一列type
    cost_level1 = cost.rename(columns={"actual_total_cost":"Actual",
                                       "budget_total_cost":"Budget",
                                       "r3m_total_cost": "R3M"})
    cost_level1 = cost_level1[['reporting_month', 'bg','cat_ind','product','cost_cat_code',"Actual","Budget", "R3M"]]
    cost_level1_melt = cost_level1.melt(['reporting_month', 'bg','cat_ind','product','cost_cat_code'])
    cost_level1_melt.rename(columns={"variable":"type"}, inplace=True)
    p_cost_level1 = cost_level1_melt.pivot_table(index=['reporting_month', 'bg','cat_ind','product','type'],
                                                 columns="cost_cat_code",values="value").reset_index()
    # Step2. 拿shipment或production出来（目标中的standard列）,计算cost/production or cost/shipment
    level1_all = pd.DataFrame()
    for standard in ["shipment", "production"]:
        # 对per shipment和per cost分别做三级汇总计算，site level,accessory/phone level, site_accessory/phone level
        result = cost_overview_level1(cost, p_cost_level1, standard)
        level1_all = pd.concat([level1_all, result])

    level1_all = add_usd_currency_and_fx_rate_for_cost_app_tables(level1_all)

    stage_to_app_indireact("cost_overview_level1", level1_all)

    """
        app.cost_overview_gap
        只用到了actual数据
        cost_overview_dtl作为底表本来就是usd单位
    """
    cost = get_table_from_app("cost_overview_dtl")
    cost = get_product_type(cost)
    cost["cat_ind"] = cost.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    cost = find_cost_code_and_non_adj(cost, left_key='cost_item_level1')
    cost["cost_cat_level0"] = cost["cost_cat_code"].apply(lambda x:x.split("_")[0])
    cost["cost_cat_level0"] = cost["cost_cat_level0"].replace({"oc":"other", "mc":"mat"})
    cost["cost_cat_level0"] = cost["cost_cat_level0"].apply(lambda x: "ucg_" + x)
    group_cost_lv0 = cost.groupby(["bg","reporting_month", "cat_ind", "cost_cat_level0"])["actual_total_cost"].sum().reset_index()
    p_lv0 = group_cost_lv0.pivot_table(index=["bg","reporting_month", "cat_ind"], columns="cost_cat_level0", values="actual_total_cost")
    p_lv0["ucg_total"] =  p_lv0.sum(axis=1)
    # non adjust actual total
    group_non_adj = cost.groupby(["bg", "reporting_month", "cat_ind", "non_adj_flag"])["actual_total_cost"].sum().reset_index().query("non_adj_flag == 1")
    group_non_adj.rename(columns={"actual_total_cost": "ucg_non_adj"}, inplace=True)

    grouped = pd.merge(p_lv0, group_non_adj[["bg", "reporting_month", "cat_ind","ucg_non_adj"]], left_index=True,
                       right_on=["bg", "reporting_month", "cat_ind"], how="left")
    grouped["type"] = "Actual"              # 这里只用到了actual

    cost_overview_gap = combine_three_cat_ind_level(grouped, standard=None)
    cost_overview_gap = add_usd_currency_and_fx_rate_for_cost_app_tables(cost_overview_gap)
    stage_to_app_indireact("cost_overview_gap", cost_overview_gap)

    """
        app.cost_overview_level2
    """
    cost = get_table_from_app("cost_overview_dtl")
    cost = get_product_type(cost)
    cost["cat_ind"] = cost.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    cost = find_cost_cat_code(cost,left_key='cost_item_level1')

    total_cost = extract_cost_or_shipment_or_production(cost, "total_cost")
    shipment = extract_cost_or_shipment_or_production(cost, "shipment")
    production = extract_cost_or_shipment_or_production(cost, "production")

    cost_and_quantity = pd.merge(total_cost, shipment, how="left",
                                 left_on=["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1','cost_item_level2',"type"],
                                 right_on=["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1','cost_item_level2',"type"])
    cost_and_quantity = pd.merge(cost_and_quantity, production, how="left",
                                 left_on=["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1','cost_item_level2',"type"],
                                 right_on=["reporting_month", 'bg', 'cat_ind', 'product','cost_cat_code','cost_item_level1','cost_item_level2',"type"])

    def tell_apart_no_level2(cost_and_quantity):
        no_level2 = cost_and_quantity.query("cost_item_level1 == cost_item_level2")     # 没有二级科目
        has_level2 = cost_and_quantity.query("cost_item_level1 != cost_item_level2")    # 有二级科目
        return no_level2, has_level2

    no_level2, has_level2 = tell_apart_no_level2(cost_and_quantity)
    no_level2["cost_item_level2"] = "aggregate"
    has_level2_agg = has_level2.groupby(['bg', "reporting_month", 'cat_ind', 'product','cost_cat_code','cost_item_level1',"type"]).sum().reset_index()
    has_level2_agg["cost_item_level2"] = "aggregate"
    cols_uniform = no_level2.columns
    concat_all = pd.concat([no_level2,has_level2[cols_uniform],has_level2_agg[cols_uniform]])

    cost_level2 = combine_three_cat_ind_level(concat_all, standard="both", groupfunc=groupby_level2_with_unit_cost)
    cost_level2.rename(columns={"product": "project",
                      "cost_item_level2": 'cost_level2'}, inplace=True)
    cost_level2 = add_usd_currency_and_fx_rate_for_cost_app_tables(cost_level2)
    stage_to_app_indireact('cost_overview_level2', cost_level2)

    """
        app.cost_prediction_dtl
    """
        # TODO(run unitcost_forecaset.py, will migrate code to here later if needed)

    """
        app.cost_prediction to app.cost_overview_level1
        在原有的cost_overview_level1基础上增加type=PredictMA, PredictES, PredictARIMA
    """

    def from_cost_prediction_dtl_to_overview_level1(predict, standard="shipment"):
        """
        :param predict: 原始的predict数据
        :param standard: 对unit_cost_shipment, unit_cost_production分别做的预测，shipment/production要分开处理
        :return: cost_overview_level1的样式数据，最终要接到该数据后面
        """

        def replace_history_actual_to_r3m(predict):
            # r3m是预测的quantity，如果是过去的月份，用actual 的quantity填充r3m的quantity，再乘以预测的unit cost，
            # 得到总的cost,因为最后要算aggregate后的unit cost,所以是sum(predict unit cost*r3m quantity)/sum(r3m quantity)
            current_date = get_latest_date_from_is_model()
            predict[f"r3m_{standard}"] = np.where(pd.to_datetime(predict["reporting_month"]) <= current_date, predict[f"actual_{standard}"], predict[f"r3m_{standard}"])
            for predict_col in prediction_cols_raw:
                predict[predict_col] = predict.eval(f"{predict_col} * r3m_{standard}")
            return predict

        ts_model_cols = ["arima", "es", "ma"]
        # get unit_cost_shipment or unit_cost_production
        prediction_cols_raw = [f"{ts_model}_unit_cost_{standard}" for ts_model in ts_model_cols]
        # change unit_cost to total cost
        predict = replace_history_actual_to_r3m(predict)

        level1_idx = ['reporting_month', 'bg', 'cat_ind', 'cost_cat_code', 'product']
        cost_use_cols = level1_idx + prediction_cols_raw
        # r3m quantity
        quantity_use_cols = ['reporting_month','product',f"r3m_{standard}"]
        predict_level1_quantity = predict[quantity_use_cols].drop_duplicates()


        # predict total cost by 3 ways
        predict_level1_cost = predict[cost_use_cols]
        # melt type and pivot level1_item
        prediction_col_renames = [f"Predict{model.split('_')[0].upper()}" for model in prediction_cols_raw] # change col name to fill type later
        rename_dict = dict(zip(prediction_cols_raw, prediction_col_renames))
        predict_level1_cost.rename(columns=rename_dict, inplace=True)
        m_predict_level1 = predict_level1_cost.melt(level1_idx)
        m_predict_level1.rename(columns={"variable":"type"}, inplace=True)
        p_predict_level1 = m_predict_level1.pivot_table(index=['reporting_month', 'bg', 'cat_ind','product',"type"],
                                                        columns="cost_cat_code", values="value").reset_index()
        # merge pivot cost with r3m quantity
        predict_level1 = p_predict_level1.merge(predict_level1_quantity,how="left",on=['reporting_month','product'])
        # cal_unit_cost_predict and aggregate by cat_ind(site,product_type,site&product type
        one_standard_predict = combine_three_cat_ind_level(predict_level1, standard=f"r3m_{standard}")
        one_standard_predict["standard"] = f"{standard}"
        return one_standard_predict

    predict = get_table_from_app("cost_prediction_dtl")
    predict = get_product_type(predict)
    predict["cat_ind"] = predict.apply(lambda x: "_".join([x["site"],x["product_type"]]), axis=1)
    predict = find_cost_cat_code(predict,left_key='cost_item_level1')

    cost_prediction_level1 = pd.DataFrame()
    for standard in ["shipment", "production"]:
        one_standard_result = from_cost_prediction_dtl_to_overview_level1(predict, standard=standard)
        cost_prediction_level1 = pd.concat([cost_prediction_level1, one_standard_result])

    cost_prediction_level1 = add_usd_currency_and_fx_rate_for_cost_app_tables(cost_prediction_level1)
    stage_to_app_indireact("cost_overview_level1", cost_prediction_level1)

