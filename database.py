from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


#sqlite
# DATABASE_URI = "sqlite:///./todos.db"
# session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# engine = create_engine(DATABASE_URI)

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="Deepa#123",
                               db="user_db"))

# Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



