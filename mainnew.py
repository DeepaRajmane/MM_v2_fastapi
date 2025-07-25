from fastapi import FastAPI, Request, Form, File, UploadFile, Depends,status,HTTPException,Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr,ValidationError
import uvicorn
from user_database import SessionLocal,get_db
from models import Users 
from fastapi.responses import RedirectResponse
import pandas as pd
from geo_spatial_data import GeoSpatialData
from random import randint
import smtplib
# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
# import pyotp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from typing import List,Optional
import sqlite3
import logging
# from market_pmap import MarketStrengthAnalyzer
from pmap_chartjs import MarketStrengthAnalyzer
import io
import xlsxwriter
from fastapi.responses import StreamingResponse


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

################# login ######################
class User(BaseModel):
   firstname:str
   lastname:str
   username:str
   email:EmailStr
   password:str
   class Config:
      orm_mode = True

@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
   return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register(request: Request):
   return templates.TemplateResponse("register.html", {"request": request})
user_db={}

@app.post('/regsubmit', response_class=HTMLResponse)
async def add_user(request: Request,fnm: str = Form(...),lnm: str = Form(...),
                    unm: str = Form(...),email: str = Form(...),pwd: str = Form(...))->User:

   
   db=SessionLocal()

   email_check = db.query(Users).filter(Users.email == email).first()   
   if email_check !=None:
      raise HTTPException(
      detail='Email is already registered',
      status_code= status.HTTP_409_CONFLICT
      )
    
   try:
      new_user=Users(firstname=fnm,lastname=lnm,username=unm,email=email, password=pwd)      
      db.add(new_user)
      db.commit()
      db.refresh(new_user) 
      db.close()
    
      redirect_url = request.url_for('login')    
      return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
   except Exception as e:
      raise HTTPException(
         status_code=status.HTTP_404_NOT_FOUND, detail=f"{e}")
   
@app.post("/loginsubmit/", response_class=HTMLResponse)
async def loginsubmit(request: Request,unm: str = Form(...),pwd: str = Form(...)):
   db=SessionLocal()
   user = db.query(Users).filter(Users.username == unm,Users.password == pwd).first()
   if not user:
      # return "Invalid Credentials"
      raise HTTPException(
         status_code=status.HTTP_404_NOT_FOUND, detail="Invalid Credentials")
   # return "login sucessful!! "
   redirect_url = request.url_for('home')    
   return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)
################################## Routes##################
# Database setup
DB_FEATURE_PATH = "features.db"
DB_DATA_PATH="data.db"

# Model for feature
class Feature(BaseModel):
    variable: str
    name: str
    category: str

# Model for data
class Data(BaseModel):
    unique_serial_number: int
    Age: int
    Gender: int
    NCCS: int
    Detailed_NCCS: int
    Townclass: int
    State: str
    City: str
    Occupation: int
    Education: int
    Online_Shopping: int
    Linear_Television: int
    Smart_Television: int
    Internet_Users: int
    Smart_Phone_Users: int
    Social_Media_Users: int
    Digital_Payment: int
    Credit_Card: int
    Netbanking: int
    Insurance: int
    Stocks_Shares: int
    Mutual_Funds: int
    Loan: int
    Electricity_Connection: int
    Ceiling_Fan: int
    LPG_Stove: int
    Two_wheeler: int
    Colour_TV: int
    Refrigerator: int
    Washing_Machine: int
    Personal_Computer_Laptop: int
    Car_Jeep_Van: int
    Air_Conditioner: int
    Year: int
    pincode: int
    latitude: float
    longitude: float


# Get all features from database
def get_features() -> List[Feature]:
    try:
        with sqlite3.connect(DB_FEATURE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT variable, name FROM features")
            features = []
            for row in cursor.fetchall():
                variable = row['variable']
                if any(var in variable for var in ['Social_Media', 'Digital_Payment', 'Credit_Card', 'Netbanking']):
                    category = "Digital Access"
                elif any(var in variable for var in ['Insurance', 'Stocks_Shares', 'Mutual_Funds', 'Loan']):
                    category = "Banking"
                else:
                    category = "Household"
                features.append(Feature(variable=variable, name=row['name'], category=category))
            logger.info(f"Retrieved {len(features)} features from database: {[f.variable for f in features]}")
            return features
    except sqlite3.Error as e:
        logger.error(f"{DB_FEATURE_PATH} Database error: {e}")
        return []

# Generate context dictionary
def generate_context(selected_features: List[str] = None,
                     selected_states: List[str] = None,
                     selected_cities: List[str] = None,
                     selected_pincodes: List[int] = None,) -> dict:
    try:
        features = get_features()
        # print(f"features:{features}")
        grouped_features = {}
        selected_features = selected_features or []
        selected_states = selected_states or []
        selected_cities = selected_cities or []
        selected_pincodes = selected_pincodes or []
        
        logger.info(f"Processing selected features: {selected_features}")
        for feature in features:
            if feature.category not in grouped_features:
                grouped_features[feature.category] = []
            checked = feature.variable in selected_features
            grouped_features[feature.category].append({
                'variable': feature.variable,
                'name': feature.name,
                'checked': checked
            })
            if checked:
                logger.debug(f"Marked feature as checked: {feature.variable}")        
        
        context = {
            'features': grouped_features,
            'nav_items': ['Home', 'Market', 'Dashboard','PerceptualMap'],  
            'selected_features': selected_features, # Pass for template debugging
            'selected_states':selected_states,
            'selected_cities': selected_cities,
            'selected_pincodes': selected_pincodes
            }
        logger.info(f"Generated context with feature categories: {grouped_features.keys()}")
        return context
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        return {'features': {}, 'nav_items': ['Home','Market','Dashboard','PerceptualMap'],
                'selected_features': [],
                'selected_states': [],
                'selected_cities':[],
                'selected_pincodes':[]                
                }

# get data from db for slected_features
def get_data():
    try:
        with sqlite3.connect(DB_DATA_PATH) as conn:            
            # Write your SQL query
            query = "SELECT * FROM data_table"# Replace with your actual table name and query
            # Load the data into a pandas DataFrame
            df = pd.read_sql_query(query, conn)
            # conn.close()
            print(f"data from data.db {df.head(2)}")            
            # Validate and collect data
            validated_data = []
            for index, row in df.iterrows():
                try:
                    record = Data(**row.to_dict())
                    validated_data.append(record)
                except ValidationError as e:
                    print(f"Validation error at row {index}:\n{e}")
            # Convert validated records back to a DataFrame
            validated_df = pd.DataFrame([record.dict() for record in validated_data])
            # print(validated_df)
            return validated_df
    except sqlite3.Error as e:
        logger.error(f"{DB_DATA_PATH} Database error: {e}")
        return []
    
################# trial data ##################

df=pd.read_csv("C:\\Users\\RajmaneD\\OneDrive - Kantar\\DR\\MM_v2_fastapi\\india_states_cities.csv")
df = df[~(df.longitude.isna() | df.latitude.isna())]
#################################
# df=get_data()

### get markets grouped by state,city and pincodes to display
markets={}
for _, row in df.iterrows():
    state = row['State']
    city = row['City']
    pincode = row['pincode']
    if state not in markets:
        markets[state] = {}
    if city not in markets[state]:
        markets[state][city] = []
    if pincode not in markets[state][city]:
        markets[state][city].append(pincode)
# print(markets)


#####################  Routes  #############################
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, selected_features: List[str] = Query(default=[]),
               selected_states: List[str] = Query(default=[]),
               selected_cities: List[str] = Query(default=[]),
               selected_pincodes: List[int] = Query(default=[])):
    context = generate_context(selected_features,selected_states,selected_cities,selected_pincodes)    
                                
    logger.info(f"Rendering home page with selected features: {selected_features},states:{selected_states},cities:{selected_cities},pincodes:{selected_pincodes}")
    return templates.TemplateResponse("home.html", {"request": request, **context,
                                                    "markets": markets})

### get percentage for selected features on selected markets
def get_percent_score(**d):
    try:        
        selected_features=d["selected_features"]
        selected_states=d["selected_states"]
        selected_cities=d["selected_cities"]
        selected_pincodes=d["selected_pincodes"]
        df_selected_pincodes = df[(df['State'].isin(selected_states)) & (df['City'].isin(selected_cities)) & (df['pincode'].isin(selected_pincodes))]
        count_selected_pincode=df_selected_pincodes[selected_features].sum().to_dict()
        perc_selected_pincodes={k: round((v/df_selected_pincodes.shape[0])*100) for k, v in count_selected_pincode.items()}
        # print(f"perc_selected_pincodes{perc_selected_pincodes}")
        return perc_selected_pincodes
    except Exception as err:
        logger.error(f"problem in get_percent_score for selected pincodes:{err}")
        print(f"problem in get_percent_score for selected pincodes:{err}")
        return {}

### get propensity score on city,state,pan india level for selected market
def get_propensity(**d):
    ## get the base to calculate propensity from json file base_propensity_score_kantar_mumbai.json for mumbai data
    with open("base_propensity_score_dummy.json", "r") as f:
        base = json.load(f)    
    perc_selected_pincodes=get_percent_score(**d)
    selected_features=d["selected_features"]
    selected_states=d["selected_states"]
    selected_cities=d["selected_cities"]    
    d_features_map=d["feature_map"]
    
    ps={}
    ps_pan_india={}
    ps_state={}
    ps_city={}
    for feature in selected_features:
        try:
            ps_pan_india[d_features_map[feature]]=round((perc_selected_pincodes[feature]/base['pan_india'][feature]*100)-100)
        except Exception as err:
            logger.error(f"problem in get_percent_score for selected pincodes:{err}")
            print(f"problem in get_propensity pan_india:{feature}-{err}")
            ps_pan_india[d_features_map[feature]]=0
        try:
            ps_state[d_features_map[feature]]=round((perc_selected_pincodes[feature]/base['state'][selected_states[0]][feature]*100)-100)
        except Exception as err:
            logger.error(f"problem in get_propensity state level:{feature}-{err}")
            print(f"problem in get_propensity state level:{feature}-{err}")
            ps_state[d_features_map[feature]]=0
        try:
            ps_city[d_features_map[feature]]=round((perc_selected_pincodes[feature]/base['city'][selected_states[0]][selected_cities[0]][feature]*100)-100)
        except Exception as err:
            ps_city[d_features_map[feature]]=0
            logger.error(f"problem in get_propensity city level:{feature}-{err}")
            print(f"problem in get_propensity city level:{feature}-{err}")
            
    ps['city']=ps_city
    ps['state']=ps_state
    ps['India']=ps_pan_india
    return ps



@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, selected_features: List[str] = Query(default=[]),
                selected_states: List[str] = Query(default=[]),
                selected_cities: List[str] = Query(default=[]),
                selected_pincodes: List[int] = Query(default=[])):

    context = generate_context(selected_features,selected_states,selected_cities,selected_pincodes)
    d_features_map={}     
    if selected_features:       
        for feature in selected_features:
            for category, features_list in context['features'].items():
                # print(f"featute list:{features_list}")
                for f in features_list:
                    # print(f)
                    if f['variable'] == feature:
                        d_features_map[f['variable']]=f['name'] 
    else:           
        for category, features_list in context['features'].items():
            for f in features_list:
                d_features_map[f['variable']]=f['name'] 
    if (selected_cities and selected_pincodes): 
        d_={"selected_features":selected_features if selected_features else list(d_features_map.keys()),
            "selected_states":selected_states,
            "selected_cities":selected_cities,
            "selected_pincodes":selected_pincodes,
            "feature_map":d_features_map}    
        ps_data = get_propensity(**d_)
        # print(ps_data)            
        try:                
            df1 = df[(df['State'].isin(selected_states)) & (df['City'].isin(selected_cities)) & (df['pincode'].isin(selected_pincodes))]               
        except Exception as e:
            logger.error(f"problem in filtering data on markets:{e}")
            print(f"problem in filtering data on markets:{e}")        
        tdf=df1[selected_features if selected_features else list(d_features_map.keys())]
    else:
        tdf=df[selected_features if selected_features else list(d_features_map.keys())]

    dist_charts = []
    for feature, values in tdf.items():
        count_1 = int(values.sum())
        count_0 = int(len(values) - count_1)
        dist_charts.append({
        "label": d_features_map[feature],
        "data": [count_1, count_0]}) 
        # print(f"dist_charts:{dist_charts}") 
          
    logger.info(f"Rendering dashboard page with selected features: {selected_features},states:{selected_states},cities:{selected_cities},pincodes:{selected_pincodes}")
    return templates.TemplateResponse("dashboard1.html", {"request": request, **context,
                                                        "markets": markets,
                                                        "charts": dist_charts,
                                                        "ps_data":ps_data if selected_pincodes else None})

                                                        

@app.get("/market", response_class=HTMLResponse)
async def market(request: Request, selected_features: List[str] = Query(default=[]),
                    selected_states: List[str] = Query(default=[]),
                    selected_cities: List[str] = Query(default=[]),
                    selected_pincodes: List[int] = Query(default=[])):
    context = generate_context(selected_features,selected_states,selected_cities,selected_pincodes)
    d_features={} 
    if selected_features:        
        for feature in selected_features:
            for category, features_list in context['features'].items():
                # print(f"featute list:{features_list}")
                for f in features_list:
                    # print(f)
                    if f['variable'] == feature:
                        d_features[f['variable']]=f['name']  
    else:           
        for category, features_list in context['features'].items():
            for f in features_list:
                d_features[f['variable']]=f['name'] 
                # print(d_features)     
    # Filter the DataFrame for selected states,cities and pincodes
    if (selected_cities and selected_pincodes):
        filtered_df = df[
        df['State'].isin(selected_states) &
        df['City'].isin(selected_cities) &
        df['pincode'].isin(selected_pincodes)
        ]
    geo_data = GeoSpatialData(filtered_df if (selected_cities and selected_pincodes) else df )
    geo_data.read_data()
    geo_data.group_by_pincode(features=selected_features if selected_features else list(d_features.keys())) 
    map_file = './static/map.html'
    geo_data.save_map(map_file,d_features)


    logger.info(f"Rendering marketmap page with selected features: {selected_features},states:{selected_states},cities:{selected_cities},pincodes:{selected_pincodes}")
    return templates.TemplateResponse("marketmap.html", {"request": request, **context,
                                                         "markets": markets})

# perceptual map display using chartjs
@app.get("/perceptualmap", response_class=HTMLResponse)
async def perceptualmap(request: Request, selected_features: List[str] = Query(default=[]),
                    selected_states: List[str] = Query(default=[]),
                    selected_cities: List[str] = Query(default=[]),
                    selected_pincodes: List[int] = Query(default=[])):
    context = generate_context(selected_features,selected_states,selected_cities,selected_pincodes)     
    if len(selected_features)>1: 
        d_selected_features_map={}       
        for feature in selected_features:
            for category, features_list in context['features'].items():            
                for f in features_list:                
                    if f['variable'] == feature:
                        d_selected_features_map[f['variable']]=f['name']
    else:
        d_features_map={}           
        for category, features_list in context['features'].items():
            for f in features_list:
                d_features_map[f['variable']]=f['name']     
    ## filtering data for selected market
    if (selected_cities and selected_pincodes):
        tdf = df[
        df['State'].isin(selected_states) &
        df['City'].isin(selected_cities) &
        df['pincode'].isin(selected_pincodes)
        ]
    else:
        tdf = df
    d={"features":selected_features if len(selected_features)>1 else list(d_features_map.keys()),
        "fmap":d_selected_features_map if len(selected_features)>1 else d_features_map,
        "df":tdf}
    msa=MarketStrengthAnalyzer(**d)
    cluster_chart_data,variance=msa.get_cluster_data()
    biplot_data = msa.get_biplot_data()
    market_strength_chart_data = msa.get_market_strength_data()
    df_mkt_strength=pd.DataFrame(market_strength_chart_data,columns=['Market','Strength'])
    print(f"market_strength_chart_data:{market_strength_chart_data}") 

    logger.info(f"Rendering settings with selected features: {selected_features}")
    return templates.TemplateResponse("perceptualmap1.html", 
                                    {"request": request, **context,
                                    "cluster_chart_data": cluster_chart_data,
                                    "variance": variance,
                                    "biplot_data": biplot_data,
                                    "market_strength_chart_data": market_strength_chart_data,
                                    "markets": markets})



@app.get("/download_excel")
async def download_excel(request: Request, selected_features: List[str] = Query(default=[]),
    selected_states: List[str] = Query(default=[]),
    selected_cities: List[str] = Query(default=[]),
    selected_pincodes: List[int] = Query(default=[])):
    selected_features = [f.strip() for f in selected_features]
    selected_states = [s.strip() for s in selected_states]
    selected_cities = [c.strip() for c in selected_cities]
    # print(f"download_excel:{selected_features},{selected_pincodes}")
   
    context = generate_context(selected_features, selected_states, selected_cities, selected_pincodes)

    if len(selected_features) > 1:
        d_selected_features_map = {}
        for feature in selected_features:
            for category, features_list in context['features'].items():
                for f in features_list:
                    if f['variable'] == feature:
                        d_selected_features_map[f['variable']] = f['name']

    if selected_cities and selected_pincodes:
        tdf = df[
            df['State'].isin(selected_states) &
            df['City'].isin(selected_cities) &
            df['pincode'].isin(selected_pincodes)
        ]
    d = {"features": selected_features ,
        "fmap": d_selected_features_map ,
        "df": tdf}
    msa = MarketStrengthAnalyzer(**d)

    # Chart data
    cluster_chart_data, variance = msa.get_cluster_data()
    biplot_data = msa.get_biplot_data()
    market_strength_chart_data = msa.get_market_strength_data()

    # Create Excel
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})

    # Sheet 1: Cluster chart
    ws1 = workbook.add_worksheet('Market Clusters')
    ws1.write_row('A1', ['PC1', 'PC2', 'Pincode'])
    for idx, point in enumerate(cluster_chart_data):
        ws1.write_row(idx + 1, 0, [point['x'], point['y'], point['label']])

    # Sheet 2: Biplot
    ws2 = workbook.add_worksheet('Biplot')
    ws2.write_row('A1', ['PC1', 'PC2', 'Label'])
    for idx, p in enumerate(biplot_data['points']):
        ws2.write_row(idx + 1, 0, [p['x'], p['y'], p['label']])
    ws2.write_row(len(biplot_data['points']) + 2, 0, ['Arrow X', 'Arrow Y', 'Feature'])
    for i, a in enumerate(biplot_data['arrows']):
        ws2.write_row(len(biplot_data['points']) + 3 + i, 0, [a['x'], a['y'], a['label']])

    # Sheet 3: Market Strength
    ws3 = workbook.add_worksheet('Market Strength')
    ws3.write_row('A1', ['Market', 'Strength'])
    for idx, (label, value) in enumerate(zip(market_strength_chart_data['labels'], market_strength_chart_data['values'])):
        ws3.write_row(idx + 1, 0, [label, round(value, 2)])

    workbook.close()
    output.seek(0)

    return StreamingResponse(
        output,
        headers={"Content-Disposition": "attachment; filename=market_analysis.xlsx"},
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# # perceptual map display using saved image
# @app.get("/perceptualmap", response_class=HTMLResponse)
# async def perceptualmap(request: Request, selected_features: List[str] = Query(default=[]),
#                     selected_states: List[str] = Query(default=[]),
#                     selected_cities: List[str] = Query(default=[]),
#                     selected_pincodes: List[int] = Query(default=[])):
#     context = generate_context(selected_features,selected_states,selected_cities,selected_pincodes)   
    
#     if len(selected_features)>1: 
#         d_selected_features_map={}       
#         for feature in selected_features:
#             for category, features_list in context['features'].items():            
#                 for f in features_list:                
#                     if f['variable'] == feature:
#                         d_selected_features_map[f['variable']]=f['name']
#     else:
#         d_features_map={}           
#         for category, features_list in context['features'].items():
#             for f in features_list:
#                 d_features_map[f['variable']]=f['name']     
#     ## filtering data for selected market
#     if (selected_cities and selected_pincodes):
#         tdf = df[
#         df['State'].isin(selected_states) &
#         df['City'].isin(selected_cities) &
#         df['pincode'].isin(selected_pincodes)
#         ]
#     else:
#         tdf = df

#     plots = []

#     d={"features":selected_features if len(selected_features)>1 else list(d_features_map.keys()),
#         "fmap":d_selected_features_map if len(selected_features)>1 else d_features_map,
#         "df":tdf}

#     msa=MarketStrengthAnalyzer(**d)
#     plots.append({"type": "Market Cluster", "data":msa.plot_market_clusters()})
#     plots.append({"type": "Cluster with Features", "data":msa.plot_biplot()})
#     plots.append({"type": "Market Strength", "data":msa.plot_market_strength()})        
#     mkt_summary_df=msa.calculate_market_strength() 
#     mkt_strength_summary=mkt_summary_df.to_dict(orient="records") 
#     columns = mkt_summary_df.columns.tolist()
   
#     logger.info(f"Rendering settings with selected features: {selected_features}")
#     return templates.TemplateResponse("perceptualmap.html", 
#                                     {"request": request, **context,
#                                     "plots": plots, 
#                                     "mkt_strength_summary":mkt_strength_summary,
#                                     "columns": columns,
#                                     "markets": markets})
 


@app.get('/signout', response_class=HTMLResponse)
async def signout(request: Request):   
   redirect_url = request.url_for('login')    
   return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)

if __name__=="__main__":
   uvicorn.run('mainnew:app',host="127.0.0.1",port = 8000,reload=True)
