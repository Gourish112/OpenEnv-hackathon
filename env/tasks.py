"""
Task definitions for the Data Cleaning & Anomaly Remediation environment.

Three tasks of increasing difficulty:
  T1 (easy)   – Type errors + obvious missing values in a small HR dataset
  T2 (medium) – Duplicates + outliers + format errors in a sales dataset
  T3 (hard)   – All of the above + hidden schema constraints + misleading hints
                in a financial transactions dataset
"""
from __future__ import annotations

from typing import Any, Dict, List
from copy import deepcopy

from env.models import (
    DataRow, DatasetSchema, DType, CellIssue
)



def _row(index: int, values: Dict[str, Any], flags: List[str] = []) -> DataRow:
    return DataRow(index=index, values=values, flags=list(flags))



T1_SCHEMA = DatasetSchema(
    columns=["employee_id", "name", "age", "salary", "is_manager"],
    dtypes={
        "employee_id": DType.INT,
        "name":        DType.STRING,
        "age":         DType.INT,
        "salary":      DType.FLOAT,
        "is_manager":  DType.BOOLEAN,
    },
    required=["employee_id", "name", "age", "salary", "is_manager"],
    unique_keys=["employee_id"],
    value_constraints={"age": {"min": 18, "max": 75}, "salary": {"min": 0.0}},
)

# Ground-truth (clean) rows
T1_GROUND_TRUTH: List[DataRow] = [
    _row(0,  {"employee_id": 1,  "name": "Alice Mbeki",    "age": 34, "salary": 72000.0, "is_manager": True}),
    _row(1,  {"employee_id": 2,  "name": "Bob Tanaka",     "age": 28, "salary": 54000.0, "is_manager": False}),
    _row(2,  {"employee_id": 3,  "name": "Chiara Russo",   "age": 41, "salary": 95000.0, "is_manager": True}),
    _row(3,  {"employee_id": 4,  "name": "David Osei",     "age": 23, "salary": 48000.0, "is_manager": False}),
    _row(4,  {"employee_id": 5,  "name": "Elif Yilmaz",    "age": 37, "salary": 61000.0, "is_manager": False}),
    _row(5,  {"employee_id": 6,  "name": "Femi Adeyemi",   "age": 55, "salary": 110000.0,"is_manager": True}),
    _row(6,  {"employee_id": 7,  "name": "Grace Kim",      "age": 29, "salary": 52000.0, "is_manager": False}),
    _row(7,  {"employee_id": 8,  "name": "Hamid Sultani",  "age": 44, "salary": 78000.0, "is_manager": False}),
    _row(8,  {"employee_id": 9,  "name": "Ingrid Larsen",  "age": 31, "salary": 67000.0, "is_manager": False}),
    _row(9,  {"employee_id": 10, "name": "Jorge Mendez",   "age": 50, "salary": 88000.0, "is_manager": True}),
]

# Dirty version exposed to the agent
T1_DIRTY: List[DataRow] = [
    _row(0,  {"employee_id": 1,  "name": "Alice Mbeki",    "age": "34",    "salary": 72000.0,  "is_manager": True},  ["wrong_type:age"]),
    _row(1,  {"employee_id": 2,  "name": "Bob Tanaka",     "age": 28,      "salary": "54000",  "is_manager": False}, ["wrong_type:salary"]),
    _row(2,  {"employee_id": 3,  "name": "Chiara Russo",   "age": 41,      "salary": 95000.0,  "is_manager": "yes"}, ["wrong_type:is_manager"]),
    _row(3,  {"employee_id": 4,  "name": "David Osei",     "age": 23,      "salary": 48000.0,  "is_manager": False}),
    _row(4,  {"employee_id": 5,  "name": "Elif Yilmaz",    "age": None,    "salary": 61000.0,  "is_manager": False}, ["missing:age"]),
    _row(5,  {"employee_id": 6,  "name": "Femi Adeyemi",   "age": 55,      "salary": 110000.0, "is_manager": True}),
    _row(6,  {"employee_id": 7,  "name": "Grace Kim",      "age": 29,      "salary": 52000.0,  "is_manager": False}),
    _row(7,  {"employee_id": 8,  "name": "Hamid Sultani",  "age": 44,      "salary": None,     "is_manager": False}, ["missing:salary"]),
    _row(8,  {"employee_id": 9,  "name": "Ingrid Larsen",  "age": 31,      "salary": 67000.0,  "is_manager": False}),
    _row(9,  {"employee_id": 10, "name": "Jorge Mendez",   "age": 150,     "salary": 88000.0,  "is_manager": True},  ["wrong_value:age"]),
]

T1_HINTS: List[CellIssue] = [
    CellIssue(row_index=0,  column="age",        issue_type="wrong_type",  current_val="34",   hint="age should be an integer"),
    CellIssue(row_index=1,  column="salary",     issue_type="wrong_type",  current_val="54000",hint="salary should be a float"),
    CellIssue(row_index=2,  column="is_manager", issue_type="wrong_type",  current_val="yes",  hint="is_manager should be boolean"),
    CellIssue(row_index=4,  column="age",        issue_type="missing",     current_val=None,   hint="age is required; this employee is 37"),
    CellIssue(row_index=7,  column="salary",     issue_type="missing",     current_val=None,   hint="salary is required; check payroll for Hamid"),
    CellIssue(row_index=9,  column="age",        issue_type="wrong_value", current_val=150,    hint="age 150 violates constraint max=75; correct value is 50"),
]

T1_TASK = {
    "task_id":          "T1_hr_type_repair",
    "difficulty":       "easy",
    "description": (
        "You are a data-quality agent for an HR system. "
        "The employee dataset below contains type errors, missing required fields, "
        "and at least one value that violates a business constraint. "
        "Repair every issue and call VALIDATE when the dataset is clean. "
        "You have full hints. Efficiency matters — every unnecessary action is penalised."
    ),
    "max_steps":        20,
    "pass_threshold":   0.75,
    "total_issues":     6,
    "schema":           T1_SCHEMA,
    "ground_truth":     T1_GROUND_TRUTH,
    "dirty_rows":       T1_DIRTY,
    "hints":            T1_HINTS,
    "visible_constraints": [
        "age must be INT in [18, 75]",
        "salary must be FLOAT >= 0",
        "is_manager must be BOOLEAN",
        "employee_id must be unique",
        "All fields are required (no nulls)",
    ],
}



T2_SCHEMA = DatasetSchema(
    columns=["order_id", "product", "region", "quantity", "unit_price", "order_date"],
    dtypes={
        "order_id":   DType.INT,
        "product":    DType.STRING,
        "region":     DType.STRING,
        "quantity":   DType.INT,
        "unit_price": DType.FLOAT,
        "order_date": DType.DATE,
    },
    required=["order_id", "product", "region", "quantity", "unit_price", "order_date"],
    unique_keys=["order_id"],
    value_constraints={
        "quantity":   {"min": 1, "max": 10000},
        "unit_price": {"min": 0.01},
    },
)

T2_GROUND_TRUTH: List[DataRow] = [
    _row(0,  {"order_id": 1001, "product": "Laptop",    "region": "North", "quantity": 5,    "unit_price": 899.99,  "order_date": "2024-01-15"}),
    _row(1,  {"order_id": 1002, "product": "Monitor",   "region": "South", "quantity": 12,   "unit_price": 349.50,  "order_date": "2024-01-16"}),
    _row(2,  {"order_id": 1003, "product": "Keyboard",  "region": "East",  "quantity": 50,   "unit_price": 79.99,   "order_date": "2024-01-17"}),
    _row(3,  {"order_id": 1004, "product": "Mouse",     "region": "West",  "quantity": 75,   "unit_price": 29.99,   "order_date": "2024-01-18"}),
    _row(4,  {"order_id": 1005, "product": "Headset",   "region": "North", "quantity": 20,   "unit_price": 129.00,  "order_date": "2024-01-19"}),
    _row(5,  {"order_id": 1006, "product": "Webcam",    "region": "South", "quantity": 30,   "unit_price": 89.99,   "order_date": "2024-01-20"}),
    _row(6,  {"order_id": 1007, "product": "Docking",   "region": "East",  "quantity": 8,    "unit_price": 199.99,  "order_date": "2024-01-21"}),
    _row(7,  {"order_id": 1008, "product": "SSD",       "region": "West",  "quantity": 100,  "unit_price": 109.99,  "order_date": "2024-01-22"}),
    _row(8,  {"order_id": 1009, "product": "RAM",       "region": "North", "quantity": 200,  "unit_price": 59.99,   "order_date": "2024-01-23"}),
    _row(9,  {"order_id": 1010, "product": "Cable",     "region": "South", "quantity": 500,  "unit_price": 9.99,    "order_date": "2024-01-24"}),
    _row(10, {"order_id": 1011, "product": "USB Hub",   "region": "East",  "quantity": 45,   "unit_price": 44.99,   "order_date": "2024-01-25"}),
    _row(11, {"order_id": 1012, "product": "Charger",   "region": "West",  "quantity": 60,   "unit_price": 39.99,   "order_date": "2024-01-26"}),
    _row(12, {"order_id": 1013, "product": "Stylus",    "region": "North", "quantity": 80,   "unit_price": 24.99,   "order_date": "2024-01-27"}),
]

# Dirty: row 13 and 14 are duplicates of row 0 and 1; row 3 has outlier quantity;
# row 6 has outlier unit_price (negative); column "rgion" instead of "region";
# order_date format error in row 8; wrong_value in row 9.
T2_DIRTY: List[DataRow] = [
    _row(0,  {"order_id": 1001, "product": "Laptop",    "rgion": "North", "quantity": 5,     "unit_price": 899.99,  "order_date": "2024-01-15"}),
    _row(1,  {"order_id": 1002, "product": "Monitor",   "rgion": "South", "quantity": 12,    "unit_price": 349.50,  "order_date": "2024-01-16"}),
    _row(2,  {"order_id": 1003, "product": "Keyboard",  "rgion": "East",  "quantity": 50,    "unit_price": 79.99,   "order_date": "2024-01-17"}),
    _row(3,  {"order_id": 1004, "product": "Mouse",     "rgion": "West",  "quantity": 99999, "unit_price": 29.99,   "order_date": "2024-01-18"}, ["outlier:quantity"]),
    _row(4,  {"order_id": 1005, "product": "Headset",   "rgion": "North", "quantity": 20,    "unit_price": 129.00,  "order_date": "2024-01-19"}),
    _row(5,  {"order_id": 1006, "product": "Webcam",    "rgion": "South", "quantity": 30,    "unit_price": 89.99,   "order_date": "2024-01-20"}),
    _row(6,  {"order_id": 1007, "product": "Docking",   "rgion": "East",  "quantity": 8,     "unit_price": -199.99, "order_date": "2024-01-21"}, ["outlier:unit_price"]),
    _row(7,  {"order_id": 1008, "product": "SSD",       "rgion": "West",  "quantity": 100,   "unit_price": 109.99,  "order_date": "2024-01-22"}),
    _row(8,  {"order_id": 1009, "product": "RAM",       "rgion": "North", "quantity": 200,   "unit_price": 59.99,   "order_date": "01/23/2024"},  ["wrong_value:order_date"]),
    _row(9,  {"order_id": 1010, "product": "Cable",     "rgion": "South", "quantity": 500,   "unit_price": 0.0,     "order_date": "2024-01-24"},  ["wrong_value:unit_price"]),
    _row(10, {"order_id": 1011, "product": "USB Hub",   "rgion": "East",  "quantity": 45,    "unit_price": 44.99,   "order_date": "2024-01-25"}),
    _row(11, {"order_id": 1012, "product": "Charger",   "rgion": "West",  "quantity": 60,    "unit_price": 39.99,   "order_date": "2024-01-26"}),
    _row(12, {"order_id": 1013, "product": "Stylus",    "rgion": "North", "quantity": 80,    "unit_price": 24.99,   "order_date": "2024-01-27"}),
    _row(13, {"order_id": 1001, "product": "Laptop",    "rgion": "North", "quantity": 5,     "unit_price": 899.99,  "order_date": "2024-01-15"}, ["duplicate"]),
    _row(14, {"order_id": 1002, "product": "Monitor",   "rgion": "South", "quantity": 12,    "unit_price": 349.50,  "order_date": "2024-01-16"}, ["duplicate"]),
]

# Partial hints – only 3 of 7 issues hinted
T2_HINTS: List[CellIssue] = [
    CellIssue(row_index=3,  column="quantity",   issue_type="outlier",     current_val=99999,    hint="quantity 99999 is statistically anomalous; expected ~75"),
    CellIssue(row_index=8,  column="order_date", issue_type="wrong_value", current_val="01/23/2024", hint="dates must be ISO-8601 YYYY-MM-DD"),
    CellIssue(row_index=13, column="order_id",   issue_type="duplicate",   current_val=1001,     hint="this row duplicates row 0"),
]

T2_TASK = {
    "task_id":          "T2_sales_multi_issue",
    "difficulty":       "medium",
    "description": (
        "You are a data-quality agent for a sales analytics pipeline. "
        "The sales order table has multiple classes of issue: a misspelled column header, "
        "outlier values, duplicate rows, an invalid date format, and a constraint violation. "
        "Only partial hints are provided. Fix all issues and call VALIDATE. "
        "Unnecessary actions reduce your score."
    ),
    "max_steps":        35,
    "pass_threshold":   0.70,
    "total_issues":     7,   # column rename + 2 outliers + 2 duplicates + date format + price=0
    "schema":           T2_SCHEMA,
    "ground_truth":     T2_GROUND_TRUTH,
    "dirty_rows":       T2_DIRTY,
    "hints":            T2_HINTS,
    "visible_constraints": [
        "quantity must be INT in [1, 10000]",
        "unit_price must be FLOAT >= 0.01",
        "order_date must be ISO-8601 (YYYY-MM-DD)",
        "order_id must be unique across all rows",
        "Column 'region' is required (check spelling)",
    ],
}



T3_SCHEMA = DatasetSchema(
    columns=["tx_id", "account_id", "tx_type", "amount", "currency",
             "merchant", "timestamp", "status"],
    dtypes={
        "tx_id":      DType.INT,
        "account_id": DType.STRING,
        "tx_type":    DType.STRING,
        "amount":     DType.FLOAT,
        "currency":   DType.STRING,
        "merchant":   DType.STRING,
        "timestamp":  DType.DATE,
        "status":     DType.STRING,
    },
    required=["tx_id", "account_id", "tx_type", "amount", "currency", "timestamp", "status"],
    unique_keys=["tx_id"],
    value_constraints={
        "amount":   {"min": 0.01},
        "tx_type":  {"enum": ["debit", "credit"]},
        "currency": {"enum": ["USD", "EUR", "GBP", "JPY"]},
        "status":   {"enum": ["pending", "completed", "failed", "reversed"]},
        "account_id": {"pattern": r"ACC-\d{4}"},
    },
)

T3_GROUND_TRUTH: List[DataRow] = [
    _row(0,  {"tx_id": 5001, "account_id": "ACC-1001", "tx_type": "debit",  "amount": 120.00, "currency": "USD", "merchant": "Amazon",     "timestamp": "2024-03-01", "status": "completed"}),
    _row(1,  {"tx_id": 5002, "account_id": "ACC-1002", "tx_type": "credit", "amount": 500.00, "currency": "EUR", "merchant": "Salary",      "timestamp": "2024-03-01", "status": "completed"}),
    _row(2,  {"tx_id": 5003, "account_id": "ACC-1003", "tx_type": "debit",  "amount": 45.50,  "currency": "GBP", "merchant": "Tesco",       "timestamp": "2024-03-02", "status": "completed"}),
    _row(3,  {"tx_id": 5004, "account_id": "ACC-1004", "tx_type": "debit",  "amount": 200.00, "currency": "USD", "merchant": "Netflix",     "timestamp": "2024-03-02", "status": "pending"}),
    _row(4,  {"tx_id": 5005, "account_id": "ACC-1005", "tx_type": "credit", "amount": 1500.00,"currency": "JPY", "merchant": "Freelance",  "timestamp": "2024-03-03", "status": "completed"}),
    _row(5,  {"tx_id": 5006, "account_id": "ACC-1001", "tx_type": "debit",  "amount": 30.00,  "currency": "USD", "merchant": "Starbucks",   "timestamp": "2024-03-03", "status": "completed"}),
    _row(6,  {"tx_id": 5007, "account_id": "ACC-1002", "tx_type": "debit",  "amount": 89.99,  "currency": "EUR", "merchant": "Spotify",     "timestamp": "2024-03-04", "status": "completed"}),
    _row(7,  {"tx_id": 5008, "account_id": "ACC-1003", "tx_type": "credit", "amount": 750.00, "currency": "GBP", "merchant": "Refund",      "timestamp": "2024-03-04", "status": "completed"}),
    _row(8,  {"tx_id": 5009, "account_id": "ACC-1004", "tx_type": "debit",  "amount": 55.00,  "currency": "USD", "merchant": "Uber",        "timestamp": "2024-03-05", "status": "completed"}),
    _row(9,  {"tx_id": 5010, "account_id": "ACC-1005", "tx_type": "debit",  "amount": 300.00, "currency": "JPY", "merchant": "Rakuten",     "timestamp": "2024-03-05", "status": "pending"}),
    _row(10, {"tx_id": 5011, "account_id": "ACC-1001", "tx_type": "credit", "amount": 2000.00,"currency": "USD", "merchant": "PayrollCo",  "timestamp": "2024-03-06", "status": "completed"}),
    _row(11, {"tx_id": 5012, "account_id": "ACC-1002", "tx_type": "debit",  "amount": 15.99,  "currency": "EUR", "merchant": "Apple",       "timestamp": "2024-03-06", "status": "completed"}),
    _row(12, {"tx_id": 5013, "account_id": "ACC-1003", "tx_type": "debit",  "amount": 400.00, "currency": "GBP", "merchant": "Rent",        "timestamp": "2024-03-07", "status": "completed"}),
    _row(13, {"tx_id": 5014, "account_id": "ACC-1004", "tx_type": "credit", "amount": 100.00, "currency": "USD", "merchant": "Cashback",    "timestamp": "2024-03-07", "status": "completed"}),
    _row(14, {"tx_id": 5015, "account_id": "ACC-1005", "tx_type": "debit",  "amount": 60.00,  "currency": "JPY", "merchant": "Convenience", "timestamp": "2024-03-08", "status": "completed"}),
]

T3_DIRTY: List[DataRow] = [
    _row(0,  {"tx_id": 5001, "account_id": "ACC-1001", "tx_type": "debit",    "amount": 120.00,  "currency": "USD", "merchant": "Amazon",     "timestamp": "2024-03-01", "status": "completed"}),
    _row(1,  {"tx_id": 5002, "account_id": "ACC-1002", "tx_type": "credit",   "amount": 500.00,  "currency": "EUR", "merchant": "Salary",      "timestamp": "2024-03-01", "status": "completed"}),
    _row(2,  {"tx_id": 5003, "account_id": "ACC1003",  "tx_type": "debit",    "amount": 45.50,   "currency": "GBP", "merchant": "Tesco",       "timestamp": "2024-03-02", "status": "completed"},  ["wrong_value:account_id"]),
    _row(3,  {"tx_id": 5004, "account_id": "ACC-1004", "tx_type": "DEBIT",    "amount": 200.00,  "currency": "USD", "merchant": "Netflix",     "timestamp": "2024-03-02", "status": "pending"},    ["wrong_value:tx_type"]),
    _row(4,  {"tx_id": 5005, "account_id": "ACC-1005", "tx_type": "credit",   "amount": 1500.00, "currency": "YEN", "merchant": "Freelance",   "timestamp": "2024-03-03", "status": "completed"},  ["wrong_value:currency"]),
    _row(5,  {"tx_id": 5006, "account_id": "ACC-1001", "tx_type": "debit",    "amount": -30.00,  "currency": "USD", "merchant": "Starbucks",   "timestamp": "2024-03-03", "status": "completed"},  ["wrong_value:amount"]),
    _row(6,  {"tx_id": 5007, "account_id": "ACC-1002", "tx_type": "debit",    "amount": 89.99,   "currency": "EUR", "merchant": "Spotify",     "timestamp": "2024-03-04", "status": "completed"}),
    _row(7,  {"tx_id": 5008, "account_id": "ACC-1003", "tx_type": "credit",   "amount": 750.00,  "currency": "GBP", "merchant": "Refund",      "timestamp": "2024-03-04", "status": "done"},       ["wrong_value:status"]),
    _row(8,  {"tx_id": 5009, "account_id": "ACC-1004", "tx_type": "debit",    "amount": 55.00,   "currency": "USD", "merchant": "Uber",        "timestamp": "2024-03-05", "status": "completed"}),
    _row(9,  {"tx_id": 5010, "account_id": "ACC-1005", "tx_type": "debit",    "amount": None,    "currency": "JPY", "merchant": "Rakuten",     "timestamp": "2024-03-05", "status": "pending"},    ["missing:amount"]),
    _row(10, {"tx_id": 5011, "account_id": "ACC-1001", "tx_type": "credit",   "amount": 2000.00, "currency": "USD", "merchant": "PayrollCo",   "timestamp": "2024-03-06", "status": "completed"}),
    _row(11, {"tx_id": 5012, "account_id": "ACC-1002", "tx_type": "debit",    "amount": "15.99", "currency": "EUR", "merchant": "Apple",       "timestamp": "2024-03-06", "status": "completed"},  ["wrong_type:amount"]),
    _row(12, {"tx_id": 5013, "account_id": "ACC-1003", "tx_type": "debit",    "amount": 400.00,  "currency": "GBP", "merchant": "Rent",        "timestamp": "2024-03-07", "status": "completed"}),
    _row(13, {"tx_id": 5014, "account_id": "ACC-1004", "tx_type": "credit",   "amount": 100.00,  "currency": "USD", "merchant": "Cashback",    "timestamp": "2024-03-07", "status": "completed"}),
    _row(14, {"tx_id": 5015, "account_id": "ACC-1005", "tx_type": "debit",    "amount": 60.00,   "currency": "JPY", "merchant": "Convenience", "timestamp": "2024-03-08", "status": "completed"}),
    # Duplicate row
    _row(15, {"tx_id": 5001, "account_id": "ACC-1001", "tx_type": "debit",    "amount": 120.00,  "currency": "USD", "merchant": "Amazon",     "timestamp": "2024-03-01", "status": "completed"},  ["duplicate"]),
    # Outlier: suspiciously large amount
    _row(16, {"tx_id": 5016, "account_id": "ACC-1004", "tx_type": "debit",    "amount": 9999999.0,"currency": "USD","merchant": "Unknown",    "timestamp": "2024-03-08", "status": "pending"},    ["outlier:amount"]),
    # Hidden cross-row issue: tx_id 5017 appears twice with different currencies
    _row(17, {"tx_id": 5017, "account_id": "ACC-1002", "tx_type": "debit",    "amount": 200.00,  "currency": "USD", "merchant": "ShopA",      "timestamp": "2024-03-09", "status": "completed"},  ["cross_row_conflict"]),
    _row(18, {"tx_id": 5017, "account_id": "ACC-1002", "tx_type": "debit",    "amount": 200.00,  "currency": "EUR", "merchant": "ShopA",      "timestamp": "2024-03-09", "status": "completed"},  ["cross_row_conflict"]),
    # Missing required field
    _row(19, {"tx_id": 5018, "account_id": "ACC-1003", "tx_type": "credit",   "amount": 75.00,   "currency": "GBP", "merchant": "Bonus",      "timestamp": None,         "status": "completed"},  ["missing:timestamp"]),
]

# MISLEADING hints: hint at row 16 says outlier is at row 14 (wrong)
T3_HINTS: List[CellIssue] = [
    CellIssue(row_index=3,  column="tx_type",   issue_type="wrong_value", current_val="DEBIT", hint="tx_type must be lowercase: 'debit' or 'credit'"),
    CellIssue(row_index=14, column="amount",    issue_type="outlier",     current_val=60.00,   hint="[MISLEADING] amount 60 looks anomalous"),   # misleading
    CellIssue(row_index=9,  column="amount",    issue_type="missing",     current_val=None,    hint="amount is required for all transactions"),
    CellIssue(row_index=15, column="tx_id",     issue_type="duplicate",   current_val=5001,    hint="tx_id 5001 already exists in row 0"),
]

T3_TASK = {
    "task_id":          "T3_financial_hard",
    "difficulty":       "hard",
    "description": (
        "You are a data-quality agent for a financial compliance pipeline. "
        "The transaction table contains type errors, missing required fields, "
        "constraint violations (invalid enums, bad account_id pattern, negative amounts), "
        "a duplicate transaction, a statistical outlier, and a hidden cross-row conflict "
        "(two rows sharing the same tx_id with different currencies — only ONE should survive). "
        "⚠️  Some hints are MISLEADING. Reason carefully before acting. "
        "Provide a 'reasoning' field in each action — it affects your score. "
        "Call VALIDATE only when you are confident the dataset is clean."
    ),
    "max_steps":        50,
    "pass_threshold":   0.65,
    "total_issues":     11,  # see flags above: 8 single-cell + 1 dup + 1 outlier + 1 cross-row
    "schema":           T3_SCHEMA,
    "ground_truth":     T3_GROUND_TRUTH,
    "dirty_rows":       T3_DIRTY,
    "hints":            T3_HINTS,
    "visible_constraints": [
        "tx_type must be 'debit' or 'credit' (lowercase)",
        "currency must be ISO-4217: USD, EUR, GBP, JPY",
        "amount must be FLOAT > 0",
        "status must be one of: pending, completed, failed, reversed",
        "account_id must match pattern ACC-NNNN",
        "tx_id must be unique",
        "All fields except merchant are required",
        "timestamp must be ISO-8601 (YYYY-MM-DD)",
        "[HIDDEN] Each tx_id may appear at most once; conflicting duplicates must be resolved",
    ],
}



TASK_REGISTRY: Dict[str, dict] = {
    "T1_hr_type_repair":  T1_TASK,
    "T2_sales_multi_issue": T2_TASK,
    "T3_financial_hard":  T3_TASK,
}


def get_task(task_id: str) -> dict:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASK_REGISTRY)}")
    return deepcopy(TASK_REGISTRY[task_id])
