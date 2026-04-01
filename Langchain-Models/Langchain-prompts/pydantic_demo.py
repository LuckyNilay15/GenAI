from pydantic import BaseModel, EmailStr
from typing import Optional
from json import dumps 

class Student(BaseModel):
    name:str
    age:Optional[int]=None
    grade:str
    email:EmailStr

new_student={'name':"Nilay",'grade':"A",'email':"nilay@gmail.com"}
student=Student(**new_student)

student_dict=dict(student)
student_json=student.model_dump_json()
print(student_dict)
print(student_json)