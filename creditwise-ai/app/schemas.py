from pydantic import BaseModel

class LoanRequest(BaseModel):
    age: int
    income: float
    previous_defaults: int
    requested_amount: float 