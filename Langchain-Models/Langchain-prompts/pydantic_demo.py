from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name:str
    age:Optional[int]=None
    grade:str

new_student={'name':'Lucky','grade':90}
student=Student(**new_student)


print(student)