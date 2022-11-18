import sqlite3

class Database:

    def __init__(self, dbpath):

        self.path = dbpath

    def add_experiment(self, dictionary):

        conn = sqlite3.connect(self.path)

        cursor = conn.cursor()
        try:
            experiment_id = cursor.execute("SELECT Experiment_id FROM Experiments ORDER BY Experiment_id DESC LIMIT 1;").fetchall()[0][0] +1
        except IndexError:
            experiment_id=1
            
        dictionary['Experiment_id'] = experiment_id
        
        columns = str(list(dictionary.keys()))[1:-1]
        values = list(dictionary.values())

        value_place_holder = ('?, '*len(values))[:-2]
        cursor.execute(f"INSERT INTO Experiments ({columns}) VALUES ({value_place_holder})", values)
        
        conn.commit()
        conn.close()

        return experiment_id

    
    def add_fov(self, Experiment_id, fov, n_lanes):

        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        
        try:
            lane_id = cursor.execute("SELECT Lane_id FROM Lanes ORDER BY Lane_id DESC LIMIT 1;").fetchall()[0][0] +1
        except IndexError:
            lane_id = 1
        

        existing = cursor.execute(f"SELECT Lane_id FROM Lanes where(Experiment_id={Experiment_id} and fov={fov});").fetchall()

        if len(existing)>0:

            cursor.execute(f"DELETE FROM LANES WHERE (Experiment_id={Experiment_id} and fov={fov});")

        for lane_index in range(n_lanes+1):

            columns = 'Lane_id, Experiment_id, fov, lane_index'
            values = f'{lane_id}, {Experiment_id}, {fov}, {lane_index}'

            cursor.execute(f"INSERT INTO Lanes ({columns}) VALUES ({values});")

            lane_id+=1

        conn.commit()
        conn.close()

        return
    
    def sql_query(self, query, path=None):

        if path is None:
            path=self.path

        conn = sqlite3.connect(path)
        conn.row_factory = lambda cursor, row: row[0]
        
        cursor = conn.cursor()

        out = cursor.execute(query).fetchall()

        conn.commit()
        conn.close()

        return out

    def add_raw_tracks(self, df,  experiment_id, fov, columns=None):

        conn = sqlite3.connect(self.path)

        df = df.rename(columns={'mass': 'nucleus_mass', 'size': 'nucleus_size', 'particle': 'particle_id', 'foot_print': 'footprint'})

        lane_ids = self.sql_query(f'SElECT lane_id from lanes where experiment_id={experiment_id} and fov={fov}')

        lane_index = self.sql_query(f'SElECT lane_index from lanes where experiment_id={experiment_id} and fov={fov}')

        lane_id_dict = dict(zip(lane_index, lane_ids))
        
        df = df[df.lane_index>0]
        df['Lane_id'] = df['lane_index'].map(lane_id_dict).astype(int)

        if columns is None:
            df = df[['frame', 'y', 'x', 'nucleus_mass', 'nucleus_size', 'particle_id',
            'nucleus', 'front', 'rear', 'footprint', 'area', 'valid',
            'interpolated', 'Lane_id', 'cyto_locator', 'FN_signal']]
        else:
            df = df[columns]

        df.to_sql(name='Raw_tracks', con = conn, if_exists='append', index=False)

        conn.commit()
        conn.close()

        return





        
        

