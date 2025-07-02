import sqlite3

# Database file path
DB_PATH = "features.db"

def create_features_table():
    try:
        # Connect to SQLite database (creates file if it doesn't exist)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Create features table with variable and name fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    variable TEXT NOT NULL,
                    name TEXT NOT NULL,
                    PRIMARY KEY (variable)
                )
            """)
            
            # Sample data provided
            features_data = [
                ("Social_Media_Users", "Social Media Users"),
                ("Digital_Payment", "Digital Payment"),
                ("Credit_Card", "Credit Card"),
                ("Netbanking", "Netbanking"),
                ("Insurance", "Insurance"),
                ("Stocks_Shares", "Stocks Shares"),
                ("Mutual_Funds", "Mutual Funds"),
                ("Loan", "Loan"),
                ("Electricity_Connection", "Electricity Connection"),
                ("Ceiling_Fan", "Ceiling Fan"),
                ("LPG_Stove", "LPG Stove"),
                ("Two_wheeler", "Two Wheeler"),
                ("Colour_TV", "Colour TV"),
                ("Refrigerator", "Refrigerator"),
                ("Washing_Machine", "Washing Machine"),
                ("Personal_Computer_Laptop", "Personal Computer Laptop"),
                ("Car_Jeep_Van", "Car Jeep Van"),
                ("Air_Conditioner", "Air Conditioner")
            ]
            
            # Check if table is empty before inserting
            cursor.execute("SELECT COUNT(*) FROM features")
            if cursor.fetchone()[0] == 0:
                # Insert the sample data
                cursor.executemany("INSERT INTO features (variable, name) VALUES (?, ?)", features_data)
                conn.commit()
                print("Features table created and populated with data successfully.")
            else:
                print("Features table already contains data.")
                
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_features_table()