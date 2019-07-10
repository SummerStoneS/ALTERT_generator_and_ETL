"""
@time: 7/5/2019 6:06 PM

@author: 柚子
"""
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from datetime import timedelta
from raw_to_staging import get_db_table_to_df
from stage_to_app import get_table_from_staging,get_table_from_app,filter_reporting_month_by_date,stage_to_app_indireact,get_latest_date_from_is_model

engine = create_engine('mssql+pymssql://localhost:1433/master',echo=True)


def get_params(param):
    params_control = get_db_table_to_df("raw", "param_control")
    param_v = params_control.query("param==@param").sort_values(by="date")["value"].values[-1]
    return param_v


def get_next_months_data(data, start_date, next=6):
    end_date =  start_date + timedelta(next*31)
    return data.query("(reporting_month<=@end_date) & (reporting_month>@start_date)")


def get_max_mps_of_next_6months(app_kpi, latest_date):
    future = get_next_months_data(app_kpi, latest_date)
    max_mps = future.groupby("product")["mps"].max().reset_index()
    max_mps.rename(columns={"mps":"max_mps"}, inplace=True)
    return max_mps

def find_line_item_short_name(data, left_key="cost_item_level1"):
    map_table = get_db_table_to_df(schema="dim", db_table_name="tbl_cost_control_list")
    merged = pd.merge(data, map_table[["line_item", 'Frontnend_name level 1']], how="left",
                      left_on=left_key, right_on="line_item")
    merged.rename(columns={"Frontnend_name level 1":"line_item_short"}, inplace=True)
    return merged

# class MohAlertGenerator:
#     def __init__(self, base_data,impact,start_alert_num, start_lead_num):
#         self.data = base_data
#         self.impact = impact
#         self.alert_num = start_alert_num
#         self.lead_num = start_lead_num
#
#     def rule7_moh_item_high_for_one_product(self,product_name):
#         single_product = self.data.query("product == @product_name")
#         if unit_moh_actual > 0.9 * unit_moh_r3m, and > 1.2 * unit_moh_quote
class AlertBase:
    alert_num = 0
    lead_num = 0
    alerts = []
    leads = []

    def __init__(self,data, impact):
        self.data = data
        self.impact = impact

    @staticmethod
    def get_alert_num():
        cls = AlertBase
        alert_id = cls.alert_num
        cls.alert_num += 1
        return alert_id

    @staticmethod
    def get_lead_num():
        cls = AlertBase
        lead_id = cls.lead_num
        cls.lead_num += 1
        return lead_id

    def get_importance(self,rule,row):
        importance = "medium"
        if row[f"rule{rule}_impact"]:
            importance = "high"
        return importance

    def run_rules(self):
        raise NotImplementedError

    def generate_alerts(self):
        raise NotImplementedError

    def get_alert_data(self, rule):
        if rule == 1:
            alert_data = self.data.query("rule1 > 0")
        elif rule == 2:
            alert_data = self.data.query("ship_is_low > 0")
        elif rule == 3:
            alert_data = self.data.query("ads_is_high > 0")
        elif rule ==4:
            alert_data = self.data.query("scrap_is_high > 0")
        elif rule == 5:
            alert_data = self.data.query("yield_rate_target_unmet > 0")
        elif rule == 6:
            alert_data = self.data.query("positive_material_cost_variance > 0")
        elif rule == 8:
            alert_data = self.data.query("rule8 > 0")
        else:
            alert_data = None
        return alert_data

    def get_description(self, rule, row):
        object = row["product"] if rule != 8 else row["site"]
        if rule == 1:
            value = "{:.2f}".format(row["production"] / row["max_capacity"] * 100)
            desc = f"{object} reporting period production volume was only {value} % of max capacity"
        elif rule == 2:
            value = "{:.2f}".format(row["rule2_num"] * 100)
            desc = f"{object} shipment volume {value} % below APR"
        elif rule == 3:
            value = "{:.2f}".format((row["ads"] / row["fg_ads_target"] - 1) * 100)
            desc = f"{object} finished goods ADS was {value} % over business rule"
        elif rule == 4:
            scrap_rate = "{:.2f}".format(row["scrap_rate"] * 100)
            scrap_target = "{:.2f}".format(get_params("scrap_target") * 100)
            diff = "{:.2f}".format((row["scrap_rate"] - get_params("scrap_target")) * 100)
            desc = f"{object} scrap rate {scrap_rate} %, {diff} % higher than {scrap_target} %"
        elif rule == 5:
            yield_rate = "{:.2f}".format(row["yield_rate"] * 100)
            diff = "{:.2f}".format((row["yield_rate_curr_target"] - row["yield_rate"]) * 100)
            desc = f"{object} yield rate was {yield_rate} %, {diff} % behind yield rate current target"
        elif rule == 6:
            material_tot_variance_over_revenue =  "{:.2f}".format(row["material_total_variance"] / row["revenue"] * 100)
            desc = f"{object} material cost variance(PPV + MUV) was {material_tot_variance_over_revenue} %"
        elif rule == 8:
            desc = []
            line_item = row['line_item_short']
            if row["tot_cost_is_high"]:
                value = "{:.2f}".format((row["actual_total_cost"] - row["budget_total_cost"]) / row["budget_total_cost"] * 100)
                desc1 = f"Actual {line_item} was {value} % higher than budget"
                desc.append(desc1)
            if row["unit_cost_is_high"]:
                unit_value = "{:.2f}".format((row["actual_unit_cost"] - row["budget_unit_cost"]) / row["budget_unit_cost"] * 100)
                desc2 = f"Actual unit {line_item} per shipment was {unit_value} % higher than budget unit cost"
                desc.append(desc2)
            desc = " ".join(desc)
        else:
            desc = ""
        return desc

    def get_leads(self, rule, row):
        object = row["product"] if rule != 8 else row["site"]
        if rule == 1:
            max_mps = round(row["max_mps"] / 1000)
            value = "{:.2f}".format(row["production"] / row["max_capacity"] * 100)
            value2 = "{:.2f}".format(max_mps / row["max_capacity"] * 100)
            production = round(row["production"] / 1000)
            lead1 = f"{object} reporting period production volume was {production}K, only {value} % of max capacity"
            lead2 = f"{object} Maximum MPS 6 month requirement was {max_mps}K, stays below {value2} % of max capacity"
            leads = [lead1, lead2]

        elif rule == 2:
            shipment = round(row["shipment"] / 1000)
            apr = round(row["apr"] / 1000)
            lead1 = f"{object} shipment volume was {shipment}K"
            lead2 = f"{object} APR was {apr}K"
            leads = [lead1, lead2]

        elif rule == 3:
            ads = "{:.2f}".format(row["ads"])
            ads_target = round(row["fg_ads_target"])
            leads = [
                f"{object} reporting month finished goods ADS {ads} days, vs. ADS business rule of {ads_target} days"]

        elif rule == 4:
            scrap_rate = "{:.2f}".format(row["scrap_rate"] * 100)
            scrap_target = "{:.2f}".format(get_params("scrap_target") * 100)
            leads = [f"{object} scrap rate {scrap_rate} %, vs.inventory reserve of {scrap_target} %"]

        elif rule == 5:
            yield_rate = "{:.2f}".format(row["yield_rate"] * 100)
            yield_rate_curr_target = "{:.2f}".format(row["yield_rate_curr_target"] * 100)
            mp_no_of_month = round(row["mp_no_of_month"])
            lead1 = f"{object} yield rate {yield_rate} %, vs. yield rate current target of {yield_rate_curr_target} %"
            lead2 = f"{object} has been in MP stage for {mp_no_of_month} months. {object} yield rate was {yield_rate}, vs. yield rate final target of {yield_rate_curr_target} %"
            leads = [lead1, lead2]

        elif rule == 6:
            ppv_revenue = "{:.2f}".format(row["ppv"] / row["revenue"] * 100)
            muv_revenue = "{:.2f}".format(row["muv"] / row["revenue"] * 100)
            leads = [f"{object} PPV was {ppv_revenue} % of revenue; MUV was {muv_revenue} % of revenue"]

        elif rule == 8:
            object = row["site"]
            line_item = row["line_item_short"]
            actual_tot = round(row["actual_total_cost"] / 1000)
            budget_tot = round(row["budget_total_cost"] / 1000)
            r3m_tot = round(row["r3m_total_cost"] / 1000)
            actual_unit = round(row["actual_unit_cost"])
            budget_unit = round(row["budget_unit_cost"])
            r3m_unit = round(row["r3m_unit_cost"])

            lead1 = f"{object} actual {line_item} was {actual_tot}K NTD, budget {line_item} was {budget_tot}K NTD"
            lead2 = f"{object} actual unit {line_item} per shipment was {actual_unit} NTD, budget unit {line_item} per shipment was {budget_unit} NTD"
            lead3 = f"{object} R3M {line_item} was {r3m_tot}K NTD; RSM unit {line_item} per shipment was {r3m_unit} NTD"
            leads = [lead1, lead2, lead3]
        else:
            leads = []
        return leads


class AlertGenerator(AlertBase):
    def __init__(self,kpi_data, impact, production_param=0.5, ads_param=1.1):
        super().__init__(kpi_data, impact)
        self.production_param = production_param
        self.ads_param = ads_param

    def rule1_production_and_max_capacity(self,param=None,impact=None):
        param = param or self.production_param
        impact = impact or self.impact
        self.data["production_is_low"] = self.data.eval("production < @param*max_capacity")
        self.data["future_is_low"] = self.data.eval("max_mps < @param*max_capacity")
        self.data["rule1"] = self.data.eval("production_is_low & future_is_low")
        self.data["rule1_num"] = self.data.eval("(max_capacity-max_mps)*fixed_moh_cost") / self.data["revenue"].sum()
        self.data["rule1_impact"] = self.data.eval("rule1_num > @impact")

    def rule2_shipment(self,impact=None):
        impact = impact or self.impact
        self.data["ship_is_low"] = self.data.eval("shipment < apr")
        self.data["rule2_num"] = self.data.eval("apr - shipment") / self.data["shipment"].sum()
        self.data["rule2_impact"] = self.data.eval("rule2_num > @impact")

    def rule3_inventory(self,param=None,impact=None):
        param = param or self.ads_param
        impact = impact or self.impact
        self.data["ads_is_high"] = self.data.eval("ads > (fg_ads_target * @param)")
        corp_funding_rate = get_params("corp_funding_rate")
        self.data["rule3_num"] = self.data.eval("(ads-fg_ads_target)*@corp_funding_rate")/self.data["revenue"].sum()
        self.data["rule3_impact"] = self.data.eval("rule3_num > @impact")

    def rule4_scrap(self,impact=None):
        impact = impact or self.impact
        scrap_target = get_params("scrap_target")
        self.data["scrap_rate"] = self.data.eval("fg_scrap_amount/revenue_amt_for_scrap")
        self.data["scrap_is_high"] = self.data.eval("scrap_rate > @scrap_target")
        self.data["rule4_impact"] = self.data.eval("(scrap_rate - @scrap_target)>@impact")

    def rule5_yield_rate(self, impact=None):
        impact = impact or self.impact
        self.data["yield_rate_target_unmet"] = self.data.eval("yield_rate < yield_rate_curr_target")
        self.data["rule5_num"] = self.data.eval("(yield_rate_curr_target - yield_rate)*cogs_for_ads/revenue")
        self.data["rule5_impact"] = self.data.eval("rule5_num > @impact")

    def rule6_positive_material_cost_variance(self, impact=None):
        impact = impact or self.impact
        self.data["positive_material_cost_variance"] = self.data.eval("material_total_variance / revenue")
        self.data["rule6_num"] = self.data["material_total_variance"] / self.data["revenue"].sum()
        self.data["rule6_impact"] = self.data.eval("rule6_num > @impact")

    def run_rules(self):
        self.rule1_production_and_max_capacity(param=self.production_param)
        self.rule2_shipment(impact=self.impact)
        self.rule3_inventory(param=self.ads_param,impact=self.impact)
        self.rule4_scrap(impact=self.impact)
        self.rule5_yield_rate(impact=self.impact)
        self.rule6_positive_material_cost_variance(impact=self.impact)
        return self.data

    def get_alert_title(self,rule):
        titles = ["Capacity under-usage","Shipment below APR","Average days stock too high","Scrap rate too high",
                  "Yield rate target unmet",'Positive material cost variance']
        rule_dict = dict(zip(range(1,7),titles))
        return rule_dict[rule]

    def generate_alerts(self):
        for rule in range(1,7):
            alert_data = self.get_alert_data(rule)
            for idx,row in alert_data.iterrows():
                alert_id = self.get_alert_num()
                title = self.get_alert_title(rule)
                alert_month = row["reporting_month"]
                object = row["product"]
                importance = self.get_importance(rule,row)
                description = self.get_description(rule, row)
                leads = self.get_leads(rule,row)
                self.alerts.append([alert_id, alert_month, title, importance, object, description])
                for lead_desc in leads:
                    lead_id = self.get_lead_num()
                    self.leads.append([alert_id, lead_id, lead_desc])

        self.alerts = pd.DataFrame(self.alerts, columns=["alert_id", "alert_month", "title","importance","object","description"])
        self.leads = pd.DataFrame(self.leads, columns=["alert_id", "lead_id", "lead_desc"])


class SiteIsAlert(AlertBase):
    def __init__(self, base, impact):
        super().__init__(base,impact)

    def cal_unit_cost(self):
        for type in ["actual", "budget", "r3m"]:
            self.data[f"{type}_unit_cost"] = self.data[f"{type}_total_cost"] / self.data[f"{type}_shipment"]

    def rule8_actual_vs_budget(self, impact=None):
        impact = impact or self.impact
        sdbg_revenue = self.data["revenue"].unique().sum()  # SDBG tot revenue
        self.data["tot_cost_is_high"] = self.data.eval("actual_total_cost > 1.2 * budget_total_cost")
        self.data["unit_cost_is_high"] = self.data.eval("actual_unit_cost > 1.2 * budget_unit_cost")
        self.data["rule8"] = self.data.eval("tot_cost_is_high | unit_cost_is_high")
        self.data["tot_cost_impact"] = self.data.eval("actual_total_cost-budget_total_cost") / sdbg_revenue
        self.data["unit_cost_impact"] = self.data.eval("(actual_unit_cost - budget_unit_cost) * actual_shipment") / sdbg_revenue
        self.data["rule8_num"] = self.data.apply(lambda x: max(x["tot_cost_impact"], x["unit_cost_impact"]), axis=1)
        self.data["rule8_impact"] = self.data.eval("rule8_num > @impact")
        self.data = self.data.query("budget_total_cost > 0")

    def run_rules(self):
        self.cal_unit_cost()
        self.rule8_actual_vs_budget()
        return self.data

    def get_alert_title(self,line_item):
        return f"{line_item} too high"

    def generate_alerts(self, rule=8):
        alert_data = self.get_alert_data(rule)
        for idx, row in alert_data.iterrows():
            alert_id = self.get_alert_num()
            title = self.get_alert_title(row["line_item_short"])
            alert_month = row["reporting_month"]
            object = row["site"]
            importance = self.get_importance(rule, row)
            description = self.get_description(rule, row)
            leads = self.get_leads(rule, row)
            self.alerts.append([alert_id, alert_month, title, importance, object, description])
            for lead_desc in leads:
                lead_id = self.get_lead_num()
                self.leads.append([alert_id, lead_id, lead_desc])

        self.alerts = pd.DataFrame(self.alerts, columns=["alert_id", "alert_month", "title", "importance", "object",
                                                         "description"])
        self.leads = pd.DataFrame(self.leads, columns=["alert_id", "lead_id", "lead_desc"])

def cal_fixed_moh_cost(data):
    # TODO(repair, property tax, administration,other没找到)
    a = data[data["moh_item"].isin([' Electricity&Water', 'other utilities','IDL','Depreciation-W-buy'])]
    fixed_moh = a.groupby(["product"])['unit_moh_actual'].sum().reset_index()
    fixed_moh.rename(columns={"unit_moh_actual":"fixed_moh_cost"}, inplace=True)
    return fixed_moh


def cal_product_ppv_muv(data, item="ppv"):
    map_dict = {"ppv": 'Material cost_Purchase price variance',
                "muv": 'Material cost_Material usage variance'}
    col_name = map_dict[item]
    ppv_data = data.query("(type == 'Actual') & (line_item == @col_name)")
    ppv = ppv_data.groupby(["product"])["value"].sum().reset_index()
    ppv.rename(columns={"value":item},inplace=True)
    return ppv


class DataPrep:
    def __init__(self, latest_date):
        self.data = get_table_from_app("perf_kpi_model")        # 各个模型各月的kpi
        self.cur_date = latest_date
        self.base_kpi = filter_reporting_month_by_date(self.data, self.cur_date)    # 各个模型当月的kpi

    def merge_models_max_mps(self):
        max_mps = get_max_mps_of_next_6months(self.data, self.cur_date)
        self.base_kpi = pd.merge(self.base_kpi, max_mps, how="left", on=["product"])

    def merge_models_moh_cost(self):
        unitmoh_monthly = get_table_from_staging("tbl_unitmoh_monthly")
        cur_month_unitmoh = filter_reporting_month_by_date(unitmoh_monthly, self.cur_date)
        fixed_moh_cost = cal_fixed_moh_cost(cur_month_unitmoh)
        self.base_kpi = pd.merge(self.base_kpi, fixed_moh_cost, how="left", on=["product"])

    def merge_mp_months_num(self):
        project_info = get_table_from_staging("dim_project")
        project_info = project_info[project_info["mp_start_month"].notnull()]
        project_info["mp_no_of_month"] = project_info["mp_start_month"].apply(
            lambda x: ((self.cur_date - pd.to_datetime(x)).days/30) if (self.cur_date - pd.to_datetime(x)).days>0 else 0)
        self.base_kpi = pd.merge(self.base_kpi, project_info[["product","mp_no_of_month"]], how="left", on=["product"])

    def merge_ppv_muv(self):
        is_modeltype = get_table_from_staging("tbl_is_modeltype")
        cur_month_is_model = filter_reporting_month_by_date(is_modeltype, self.cur_date)
        for item in ["ppv", "muv"]:
            ppv = cal_product_ppv_muv(cur_month_is_model, item=item)
            self.base_kpi = pd.merge(self.base_kpi, ppv, how="left", on=["product"])

    def prep_init(self):
        self.merge_models_max_mps()
        self.merge_models_moh_cost()
        self.merge_mp_months_num()
        self.merge_ppv_muv()

def merge_unitmoh(base_kpi):
    stage_unitmoh_monthly = get_table_from_staging("tbl_unitmoh_monthly")
    cur_month_unitmoh = filter_reporting_month_by_date(stage_unitmoh_monthly, latest_date)
    product_unitmoh = pd.merge(base_kpi[["reporting_month", "product", "production"]],
                               cur_month_unitmoh[["product","moh_item", "unit_moh_actual", "unit_moh_r3m", "unit_moh_quote"]],
                               how="left", on=["product"])
    return product_unitmoh


def merge_is_model(base_kpi):
    app_is = get_table_from_app("cost_overview_dtl")
    cur_month_is = filter_reporting_month_by_date(app_is, latest_date)
    cur_month_is = cur_month_is[cur_month_is["cost_box_group"].isin(['Other cost', 'OPEX'])]
    # from usd to ntd
    for type in ["actual", "budget", "r3m"]:
        cur_month_is[f"{type}_total_cost"] = cur_month_is[f"{type}_total_cost"] * cur_month_is["fx_rate"]
    level1_group = cur_month_is.groupby(['reporting_month','product','cost_item_level1']).\
        agg({'actual_total_cost':'sum',
             'budget_total_cost':"sum",
             'r3m_total_cost': "sum",
             'actual_shipment': "mean",
             'budget_shipment':"mean",
             'r3m_shipment': "mean"
             }).reset_index()
    cur_is = find_line_item_short_name(level1_group)
    base_is = pd.merge(base_kpi[["reporting_month", 'site', "product", "revenue"]],
                       cur_is[["product", 'line_item_short','actual_total_cost','actual_shipment', 'budget_total_cost',
                               'budget_shipment', 'r3m_total_cost', 'r3m_shipment']],
                       how="left", on=["product"])
    return base_is


if __name__ == '__main__':
    # get current month
    latest_date = get_latest_date_from_is_model()
    # prepare kpi data input for kpi alert
    data_processor = DataPrep(latest_date)
    data_processor.prep_init()
    base_input = data_processor.base_kpi
    # generate alert for kpi table
    KpiAlertMaker = AlertGenerator(base_input, impact=0.00001, production_param=0.5, ads_param=1.1)
    kpi_alert_source = KpiAlertMaker.run_rules()
    KpiAlertMaker.generate_alerts()
    """
        source是staging.tbl_is_modeltype的alert
    """
    is_alert_base = merge_is_model(base_input)              # by model level
    site_alert_base = is_alert_base.groupby(["reporting_month", "site", "line_item_short"]).sum().reset_index()
    SiteAlertMaker = SiteIsAlert(site_alert_base, impact=0.00001)
    site_alert_source = SiteAlertMaker.run_rules()
    SiteAlertMaker.generate_alerts(rule=8)

    # insert result into db
    stage_to_app_indireact("perf_alerts", SiteAlertMaker.alerts, db_id=False)      # app表中没有id
    stage_to_app_indireact("perf_alert_leads", SiteAlertMaker.leads, db_id=False)  # app表中没有id


    # SiteAlertMaker.alerts.to_excel("perf_alerts.xlsx")
    # SiteAlertMaker.leads.to_excel("perf_alert_leads.xlsx")
    # site_alert_source.to_excel("site_is_alert_source.xlsx")

    # insert middle table into db staging table
    # kpi_alert_source = kpi_alert_source.replace(np.inf,np.nan)
    # kpi_alert_source = kpi_alert_source.replace(-np.inf, np.nan)
    # kpi_alert_source.to_sql("alert_tag", con=engine, schema="staging", if_exists="append", index=False)  # 原始数据

    # """
    #     source是staging.tbl_unitmoh_monthly的alert
    # """
    # # get base input
    # unitmoh_alert_base = merge_unitmoh(base_input)




