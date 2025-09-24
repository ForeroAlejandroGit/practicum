import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import present_value as PresentValue

class EDA:
    def __init__(self, filename: str):
        self.filename = filename
    
    def get_head(self, df: pd.DataFrame) -> pd.DataFrame:
        df_head = df.iloc[ 0:15 , 0:2 ]
        df_head = pd.DataFrame([df_head.iloc[:,1].to_list()], columns=df_head.iloc[:,0].to_list())

        # Assign dtypes: categorical, string, and numeric
        str_cols = [ "NOMBRE DEL PROYECTO"]
        int_cols = [ "AÑO INICIO"]

        df_head[str_cols] = df_head[str_cols].astype("string")
        df_head[int_cols] = df_head[int_cols].astype("Int64")
        
        return df_head.loc[:, ['NOMBRE DEL PROYECTO','AÑO INICIO']]

    def get_uf(self, df: pd.DataFrame) -> pd.DataFrame:
        df_uf = df.iloc[0:11,3:].iloc[ : , :-1 ]
        column_names = (df_uf.iloc[1:, 0].astype(str) + " " + df_uf.iloc[1:, 1].astype(str)).to_list()
        column_names = [column.replace(" nan", "") for column in column_names]
        df_uf = df_uf.drop(df_uf.columns[[1]], axis=1)
        
        return df_uf, column_names

    def get_items(self, df: pd.DataFrame) -> pd.DataFrame:
        
        columns_names_items = [ "1 - TRANSPORTE", "2 - TRAZADO Y DISEÑO GEOMÉTRICO", "2.1 - INFORMACIÓN GEOGRÁFICA", "2.2 TRAZADO Y DISEÑO GEOMÉTRICO", 
                            "2.3 - SEGURIDAD VIAL", "2.4 - SISTEMAS INTELIGENTES", "3 - GEOLOGÍA", "3.1 - GEOLOGÍA", "3.2 - HIDROGEOLOGÍA", 
                            "4 - SUELOS", "5 - TALUDES", "6 - PAVIMENTO", "7 - SOCAVACIÓN", "8 - ESTRUCTURAS", "9 - TÚNELES", "10 - URBANISMO Y PAISAJISMO", 
                            "11 - PREDIAL", "12 - IMPACTO AMBIENTAL", "13 - CANTIDADES", "14 - EVALUACIÓN SOCIOECONÓMICA", "15 - OTROS - MANEJO DE REDES" ]
        
        df_items = df.iloc[ 17:, 0:2 ]
        df_items = pd.DataFrame([df_items.iloc[:,1].to_list()], columns=columns_names_items) 
        return df_items
    
    def assemble_sheet(self, df: pd.DataFrame) -> pd.DataFrame:

        df_head = self.get_head(df)
        df_uf, column_names = self.get_uf(df)
        df_items = self.get_items(df)
        rows = []
        
        #Create a row for each functional unit
        for i in range(1, df_uf.shape[1]):
            
            #Aggregate longitud, puentes, tuneles for the current functional unit
            df_uf_x = pd.DataFrame([df_uf.iloc[1:,i].to_list()], columns=column_names)  
            df_uf_x['NOMBRE UF'] = df_uf.iloc[0, i]
            
            df_items_for_functional_unit = df_items / 1 # Future consideration divide  by df_uf_totals
            
            row = pd.concat([df_head, df_uf_x, df_items_for_functional_unit], axis=1)
            rows.append(row)
            
        return pd.concat(rows, axis=0, ignore_index=True)
    
    def assemble_project(self) -> pd.DataFrame:
        with pd.ExcelFile(self.filename, engine="openpyxl") as xls:
            
            project_names = [project_name for project_name in xls.sheet_names if project_name.isnumeric()]
            df_project =[]

            for project_name in project_names:
                df = pd.read_excel(self.filename, sheet_name=project_name, header=None, engine="openpyxl")
                df_project.append(self.assemble_sheet(df))
                #TEMPORAL DEBUGGING
                if project_name == '45000036221':
                    break 

        return pd.concat(df_project, axis=0, ignore_index=True)
    
    def weighted_values(self, row: pd.Series) -> pd.Series:

        row = row.fillna(0)

        #Longitude analysis
        longitude_weigth = row['LONGITUD KM WEIGHT']
        row['1 - TRANSPORTE'] *= longitude_weigth
        row['2 - TRAZADO Y DISEÑO GEOMÉTRICO'] *= longitude_weigth
        row['2.1 - INFORMACIÓN GEOGRÁFICA'] *= longitude_weigth
        row['2.2 TRAZADO Y DISEÑO GEOMÉTRICO'] *= longitude_weigth
        row['2.3 - SEGURIDAD VIAL'] *= longitude_weigth
        row['2.4 - SISTEMAS INTELIGENTES'] *= longitude_weigth
        row['3 - GEOLOGÍA'] *= longitude_weigth   
        row['3.1 - GEOLOGÍA'] *= longitude_weigth
        row['3.2 - HIDROGEOLOGÍA'] *= longitude_weigth

        row['5 - TALUDES'] *= longitude_weigth
        row['6 - PAVIMENTO'] *= longitude_weigth
        row['7 - SOCAVACIÓN'] *=     longitude_weigth

        row['11 - PREDIAL'] *= longitude_weigth
        row['12 - IMPACTO AMBIENTAL'] *= longitude_weigth

        row['15 - OTROS - MANEJO DE REDES'] *= longitude_weigth
        
        #Bridge analysis
        bridge_weigth = 1
        if row['PUENTES VEHICULARES UND'] > 0 or row['PUENTES PEATONALES UND'] > 0:
            bridges_ratio = 3
            bridge_weigth = ((row['PUENTES VEHICULARES UND WEIGHT'] + row['PUENTES VEHICULARES M2 WEIGHT'])*bridges_ratio + row['PUENTES PEATONALES UND WEIGHT'])/bridges_ratio*3
            row['4 - SUELOS'] *= bridge_weigth
            row['8 - ESTRUCTURAS'] *= bridge_weigth
        
        #Tunnel analysis
        tunnel_weight = 1
        if row['TUNELES UND'] > 0:
            tunnel_weight = row['TUNELES UND WEIGHT'] + row['TUNELES M2 WEIGHT']
            row['9 - TÚNELES'] *= tunnel_weight
        
        #Urbanism analysis
        urbanism_weight = 1
        if row['PUENTES PEATONALES UND'] > 0:  
            urbanism_weight = row['PUENTES PEATONALES UND WEIGHT']
            row['10 - URBANISMO Y PAISAJISMO'] *= urbanism_weight
        
        return row

    def create_dataset(self, present_value_costs) -> pd.DataFrame:
        
        df = self.assemble_project()

        mask = df.columns[df.columns.str.match(r"^\d")].tolist()
        df_present_value = df.apply(present_value_costs, axis=1, mask=mask, present_year=2025)
        df = df_present_value.drop(columns=['AÑO INICIO', 'NOMBRE UF'])

        cols = df.loc[:, 'LONGITUD KM':'TUNELES M2'].columns
        totals = df.groupby('NOMBRE DEL PROYECTO')[cols].transform('sum').replace(0, pd.NA)
        w = (df[cols] / totals).fillna(0)
        w.columns = [f'{c} WEIGHT' for c in cols]
        df = df.join(w)
        df =df.apply(self.weighted_values, axis=1)
        df = df.drop(columns=['NOMBRE DEL PROYECTO'])
        df = df.loc[:, 'LONGITUD KM':'15 - OTROS - MANEJO DE REDES']
        
        return df