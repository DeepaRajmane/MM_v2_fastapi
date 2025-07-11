from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from user_database import Base,engine


class Users(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, nullable=False,index=True)
    firstname = Column(String(50))
    lastname = Column(String(50))
    username = Column(String(50), unique=True)
    email = Column(String(50), unique=True)
    password = Column(String(50))    

# Create the tables in the database

Base.metadata.create_all(bind=engine)