import requests, pandas as pd, time

AUTH_URL = "https://gestiona.ingetec.com.co/ReportBi/Auth/login"
DATA_URL = "https://gestiona.ingetec.com.co/ReportBi/api/Report/GetSalaryIncreasePerYear"
USERNAME = "powerbi_report"
PASSWORD = "1ngete2025_*"

class PresentValue:

    def __init__(self):
        self.incremento = None
       
    def isEmpty(self) -> bool:
        if self.incremento is None:
            return True
        return False
        
    def fetch_salary_increase_per_year(self) -> pd.Series:
        """Authenticate and fetch the yearly salary increases (list of dicts)."""
        """Fetch increments and cache for `cache_ttl` each month in seconds."""
        if self.incremento is not None:
            return self.incremento  # return cached data
        
        with requests.Session() as s:
            auth = s.post(AUTH_URL, json={"username": USERNAME, "password": PASSWORD}, timeout=15)
            auth.raise_for_status()
            token = auth.json().get("token") or auth.json().get("access_token")
            if not token:
                raise RuntimeError("No token found in auth response.")
            resp = s.get(DATA_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            resp.raise_for_status()
            
            data = resp.json()
            self.incremento = pd.Series({item["año_aumento"]: item["incremento"] for item in data})
            
            return self.incremento

    def present_value(self, past_value: float, past_year: int, present_year: int = None) -> float:
        """Compound yearly increments from past_year+1 to present_year and return present value."""
        inc = self.incremento.copy().astype(float)
        inc.index = inc.index.astype(int)
        if inc.max() > 1.0:
            inc = inc / 100.0

        if present_year is None:
            present_year = min(pd.Timestamp.now().year, int(inc.index.max()))
        if present_year <= past_year:
            return float(past_value)

        years = range(past_year + 1, present_year + 1)
        inc_years = inc.reindex(years).fillna(0.0)
        factor = (1.0 + inc_years).prod()
        return float(past_value) * float(factor)
    
    def present_value_costs(self, row: pd.Series, mask: list[str], present_year: int) -> pd.Series:
        for col in mask:
            row[col] = self.present_value(row[col], row['AÑO INICIO'], present_year) 
        return row

    
# Example: compute present value for 1,000,000 from 2015 to latest available
# example_value = 1_000_000
# example_past_year = 2015
# present_val = present_value(example_value, example_past_year)
# present_val
