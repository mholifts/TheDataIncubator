#!/usr/bin/env python
# coding: utf-8

# # **AI driven MBS Trading**

# # **The primary driver of value in Agency Mortgage Backed Securities is prepayments**

# # Prepayments are largely driven by a borrower's incentive to refinance

# # ... other drivers like the size of the loan are also important

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
from datadriven import SqlQuery, list_fields, execute_query, list_tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table_name = "fixed_fh_fn_results"
schema = "fixed"

filters = "report_month >= '2019-08-01'"          " and product_pool in ('FNM30', 'FHLG30')"          " and coupon_pool between 2.5 and 12"          " and spec_pool_type in ('LLB', 'MLB', 'HLB', '200K', 'TBA')"
hide_small_buckets = True
small_buckets_size = 3000
incentive_range = [-0.75, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5]
wala_range = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 18, 24, 36, 48, 90, 120, 180, 240, 360]
eps = 1e-10

filter_wala = " wala between 15 and 89 "

query = SqlQuery("{schema}.{table} where {filters} and {filter_wala}".format(schema=schema, table=table_name, filters=filters, filter_wala=filter_wala))
query.create_field_range("incentive_rng", "incentive", incentive_range)
query.select("incentive_rng is not null")
query.create_tabulation(["spec_pool_type", "incentive_rng"], {"smm" : ("avg(cls)", "smm"), "loan_number" : ("sum", "loan_number")})
query.create_field("cpr", "smm_to_cpr(smm)")
df = execute_query(query)
table = pd.pivot_table(df, values='cpr', index=['incentive_rng'], columns=['spec_pool_type'], aggfunc=np.sum)


# In[21]:


table.plot(title="S-Curves by loan size", xlabel="Incentive to refinance",ylabel="Annual prepayments (CPR)")


# # Objective of my capstone is to predict prepayments on bonds by doing ML on monthly loan level data

# # Lots of fields to look at..

# In[22]:


list_fields ("fixed_fh_fn_results")[['name','description']]


# # If possible, provide insight into what drivers are impacting prepayment projections the most
