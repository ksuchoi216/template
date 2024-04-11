import pandas as pd
import numpy as np
import holidays


class TimeCovariates:
    def __init__(self, df_dt, normalized=True):
        self.normalized = normalized
        self.dt = pd.to_datetime(df_dt)

    def get_covariates(self):
        month = np.array(self.dt.dt.month, dtype=np.float32)
        if self.normalized:
            month = month / 11.0 - 0.5
        dayofyear = np.array(self.dt.dt.dayofyear, dtype=np.float32)
        if self.normalized:
            dayofyear = dayofyear / 364.0 - 0.5
        dayofweek = np.array(self.dt.dt.dayofweek, dtype=np.float32)
        if self.normalized:
            dayofweek = dayofweek / 6.0 - 0.5
        hour = np.array(self.dt.dt.hour, dtype=np.float32)
        if self.normalized:
            hour = hour / 23.0 - 0.5
        minute = np.array(self.dt.dt.minute, dtype=np.float32) // 15
        if self.normalized:
            minute = minute / 59.0 - 0.5

        cos_time = np.cos(2 * np.pi * (self.dt.dt.hour / 24))
        sin_time = np.sin(2 * np.pi * (self.dt.dt.hour / 24))

        # df["cos_time"] = np.cos(2 * np.pi * (df["dt"].hour / 24))
        # df["sin_time"] = np.sin(2 * np.pi * (df["dt"].hour / 24))
        all_covs = np.column_stack(
            [month, dayofyear, dayofweek, hour, minute, cos_time, sin_time]
        )

        columns = [
            "month",
            "dayofyear",
            "dayofweek",
            "hour",
            "minute",
            "cos_time",
            "sin_time",
        ]
        dft = pd.DataFrame(data=all_covs, columns=columns, index=self.dt)
        return dft


class TimeCovariatesFull:
    def __init__(self, df_dt, normalized=True, selected_cols=None):
        # df = df.reset_index()
        self.dt = pd.to_datetime(df_dt)
        self.national_holidays = holidays.KR()
        self.normalized = normalized
        self.selected_cols = selected_cols

    def _minute_of_hour(self):
        minutes = np.array(self.dt.dt.minute, dtype=np.float32)
        if self.normalized:
            minutes = minutes / 59.0 - 0.5
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dt.dt.hour, dtype=np.float32)
        if self.normalized:
            hours = hours / 23.0 - 0.5
        return hours

    def _day_of_week(self):
        day_week = np.array(self.dt.dt.dayofweek, dtype=np.float32)
        if self.normalized:
            day_week = day_week / 6.0 - 0.5
        return day_week

    def _day_of_month(self):
        day_month = np.array(self.dt.dt.day, dtype=np.float32)
        if self.normalized:
            day_month = day_month / 30.0 - 0.5
        return day_month

    def _day_of_year(self):
        day_year = np.array(self.dt.dt.dayofyear, dtype=np.float32)
        if self.normalized:
            day_year = day_year / 364.0 - 0.5
        return day_year

    def _month_of_year(self):
        month_year = np.array(self.dt.dt.month, dtype=np.float32)
        if self.normalized:
            month_year = month_year / 11.0 - 0.5
        return month_year

    def _week_of_year(self):
        week_year = np.array(self.dt.dt.strftime("%U").astype(int), dtype=np.float32)
        if self.normalized:
            week_year = week_year / 51.0 - 0.5
        return week_year

    def _weekends(self):
        dow = self._day_of_week().reshape(1, -1)
        weekends = np.where(dow >= 5, 1, 0)
        if self.normalized:
            weekends = weekends / 1 - 0.5
        # print(weekends.shape)
        return weekends.reshape(-1)

    def _national_holidays(self):
        n_holidays = np.array(
            [1 if x in self.national_holidays else 0 for x in self.dt]
        )
        if self.normalized:
            n_holidays = n_holidays / 1 - 0.5
        return n_holidays.reshape(-1)

    def get_covariates(self):
        moh = self._minute_of_hour()
        hod = self._hour_of_day()
        dom = self._day_of_month()
        dow = self._day_of_week()
        doy = self._day_of_year()
        moy = self._month_of_year()
        woy = self._week_of_year()
        wend = self._weekends()
        nhod = self._national_holidays()

        # print(moh.shape, hod.shape, wend.shape)
        all_covs = np.column_stack(
            [
                moh,
                hod,
                dom,
                dow,
                doy,
                moy,
                woy,
                wend,
                nhod,
            ]
        )

        columns = ["moh", "hod", "dom", "dow", "doy", "moy", "woy", "wend", "nhod"]
        dft = pd.DataFrame(data=all_covs, columns=columns, index=self.dt)

        if self.selected_cols is not None:
            dft = dft[self.selected_cols]

        return dft
