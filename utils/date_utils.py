# In a new file: utils/date_utils.py
from datetime import datetime, timedelta

def get_date_range(scope: str):
    """
    Calculate date range based on scope.
    Args:
        scope: one of ["this_week", "this_month", "this_quarter", "this_year"]
    Returns:
        tuple of (start_date, end_date)
    """
    today = datetime.now().date()
    year_start = datetime(today.year, 1, 1).date()
    
    if scope == "this_week":
        # Start from Monday of current week
        start_date = today - timedelta(days=today.weekday())
        return start_date, today
        
    elif scope == "this_month":
        # Start from first day of current month
        start_date = today.replace(day=1)
        return start_date, today
        
    elif scope == "this_quarter":
        # Start from first day of current quarter
        quarter = (today.month - 1) // 3
        start_date = today.replace(
            month=3 * quarter + 1,
            day=1
        )
        return start_date, today
        
    elif scope == "last_month":
        # Last month's date range
        if today.month == 1:
            start_date = today.replace(year=today.year-1, month=12, day=1)
        else:
            start_date = today.replace(month=today.month-1, day=1)
        end_date = today.replace(day=1) - timedelta(days=1)
        return start_date, end_date
        
    elif scope == "last_quarter":
        # Last quarter's date range
        quarter = (today.month - 1) // 3
        if quarter == 0:
            start_date = today.replace(year=today.year-1, month=10, day=1)
            end_date = today.replace(year=today.year-1, month=12, day=31)
        else:
            start_date = today.replace(month=3 * (quarter - 1) + 1, day=1)
            end_date = today.replace(month=3 * quarter, day=1) - timedelta(days=1)
        return start_date, end_date
        
    elif scope == "last_year":
        # Last year's date range
        start_date = today.replace(year=today.year-1, month=1, day=1)
        end_date = today.replace(year=today.year-1, month=12, day=31)
        return start_date, end_date
        
    elif scope == "ytd" or scope == "this_year":
        # Year to date
        return year_start, today
        
    else:
        # Default to last 30 days if scope not recognized
        start_date = today - timedelta(days=30)
        return start_date, today