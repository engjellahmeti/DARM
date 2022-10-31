# @project Deviance Analysis by Means of Redescription Mining - Master Thesis 
# @author Engjëll Ahmeti
# @date 11/22/2020

import pyodbc
import datetime as dt
from event_log_reader.reader import EventLogReader
import re
import pandas as pd
from feature_vectors.declare_constraint import DeclareConstraint
from log_print import Print
from typing import List
import os

class RXESApproach:
    def __init__(self):
        self.connection_string = r'DRIVER={SQL Server};SERVER=engjell-pc\unibayreuth;DATABASE=EventLog;'

    def insert_logs(self, event_log_name):
        try:
            cnxn = pyodbc.connect(self.connection_string)
            cursor = cnxn.cursor()
            Print.YELLOW.print('Inserting log in the DB.')

            sql = "SET NOCOUNT ON; declare @LogId int; exec InsertLog @LogName=?, @LogId=@LogId output; select @LogId;"
            params = (event_log_name)
            cursor.execute(sql, params)
            logId = cursor.fetchall()[0]
            cursor.commit()
            if logId:
                return logId[0]

            cnxn.close()
            return -1

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return -1

    def insert_traces(self, traces, logId):
        try:
            cnxn = pyodbc.connect(self.connection_string)
            cursor = cnxn.cursor()
            Print.YELLOW.print('Inserting traces of the log in the DB.')

            for concept_name in traces:
                trace = concept_name['concept:name']
                if trace != '':
                    sql = "SET NOCOUNT ON; exec InsertTraces @logId=?, @trace=?"
                    params = (logId, str(logId) +'-'+ trace)
                    cursor.execute(sql, params)
                    cursor.commit()

            cnxn.close()

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()

    def insert_events(self, cases, logId):
        try:
            cnxn = pyodbc.connect(self.connection_string)
            cursor = cnxn.cursor()
            Print.YELLOW.print('Inserting events of the log in the DB.')

            for case in cases.keys():
                for event in case:
                    trace = event.caseID
                    activity = re.sub(' ', '', event.activityName)
                    timestamp = event.timestamp
                    timestamp = dt.datetime.strptime(re.sub(r'\+\d*:\d*', '', timestamp), '%Y-%m-%dT%H:%M:%S.%f')

                    sql = "SET NOCOUNT ON; DECLARE @eventId int; exec CreateEvent @logId=?, @trace = ?, @ConecptName = ?, @Timestamp=?, @eventId=@eventId output; select @eventId"

                    timestamp = event['time:timestamp'][1]
                    if logId == 4:
                        timestamp = dt.datetime.strptime(re.sub(r'\+\d*:\d*', '', timestamp), '%Y-%m-%dT%H:%M:%S')
                    else:
                        timestamp = dt.datetime.strptime(re.sub(r'\+\d*:\d*', '', timestamp), '%Y-%m-%dT%H:%M:%S.%f')

                    params = (str(logId) +'-'+ trace, trace, activity, timestamp)
                    cursor.execute(sql, params)
                    eventId = cursor.fetchall()[0]
                    cursor.commit()
                    if eventId:
                        self.insert_attributes(event.payload, eventId[0])

            cnxn.close()

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()

    def insert_attributes(self, attributes, eventId):
        try:
            cnxn = pyodbc.connect(self.connection_string)
            cursor = cnxn.cursor()

            for attributeKey in attributes:
                sql = "SET NOCOUNT ON; exec CreateAttribute @eventId=?, @attributeKey = ?, @attributeType=?, @attributeValue=?"
                params = (eventId, attributeKey, 'string', attributes[attributeKey])
                cursor.execute(sql, params)
                cursor.commit()

            cnxn.close()

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()

    def rxes(self, event_log_name, traces, cases):
        #insert log
        logId = self.insert_logs(event_log_name)

        if logId != -1:
            #insert traces
            self.insert_traces(traces, logId)

            #insert activities
            self.insert_events(cases, logId)

            Print.YELLOW.print('The event log was successfully inserted.')

        else:
            Print.RED.print('The event log was not inserted')

        return logId

    def rxes(self, file_path):
        nn = file_path.split('\\')

        event_log_name = re.search(r'([^\.]*)\.xes', nn[len(nn) -1], re.S|re.I)
        if event_log_name:
            event_log_name = event_log_name.group(1)
            event_log_reader = EventLogReader()

            (traces, cases) = event_log_reader.gain_log_info_table(file_path=file_path)


            #insert log
            logId = self.insert_logs(event_log_name)

            if logId != -1:
                #insert traces
                self.insert_traces(traces, logId)

                #insert activities
                self.insert_events(cases, logId)
                Print.YELLOW.print('The event log was successfully inserted.')

            else:
                Print.RED.print('The event log was not inserted')

            return logId

    def mine_constraints(self, filename, log_id, no_of_rows) -> List[DeclareConstraint]:
        mined_constraints: List[DeclareConstraint] = []
        mined_constraints = mined_constraints + self.get_response_constraints(log_id=log_id, no_of_rows=no_of_rows)
        mined_constraints = mined_constraints + self.get_chain_response_constraints(log_id=log_id, no_of_rows=no_of_rows)
        mined_constraints = mined_constraints + self.get_alternate_response_constraints(log_id=log_id, no_of_rows=no_of_rows)
        mined_constraints = mined_constraints + self.get_precedence_constraints(log_id=log_id, no_of_rows=no_of_rows)
        mined_constraints = mined_constraints + self.get_chain_precedence_constraints(log_id=log_id, no_of_rows=no_of_rows)
        mined_constraints = mined_constraints + self.get_alternate_precedence_constraints(log_id=log_id, no_of_rows=no_of_rows)

        path = os.path.abspath('rxes_approach/mined_constraints/' + filename +'.decl')
        with open(path, 'wt') as a:
            a.write("Constraint,Activitation,Target\n")
        
            for constraint in mined_constraints:
                a.write("{0},{1},{2}\n".format(constraint.rule_type, constraint.activation, constraint.target))
        
        return mined_constraints            

    def get_response_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetResponseDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining response constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['A'], target=row['B'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None
    
    def get_chain_response_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetChainResponseDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining chain response constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['A'], target=row['B'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None
    
    def get_alternate_response_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetAlternateResponseDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining alternate response constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['A'], target=row['B'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None

    def get_precedence_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetPrecedenceDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining precedence constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['B'], target=row['A'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None

    def get_chain_precedence_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetChainPrecedenceDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining chain precedence constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['B'], target=row['A'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None

    def get_alternate_precedence_constraints(self, log_id, no_of_rows) -> List[DeclareConstraint]:
        try:
            cnxn = pyodbc.connect(self.connection_string)
            sql = "SET NOCOUNT ON; exec GetAlternatePrecedenceDeclareConstraints @logID={0}, @noOfRows={1}".format(log_id, no_of_rows)
            Print.YELLOW.print('Mining alternate precedence constraints from the DB.')

            declare_constraints_frame = pd.read_sql(sql, cnxn)

            declare_constraints: List[DeclareConstraint] = []
            for row in declare_constraints_frame.iterrows():
                row = row[1]
                dc = DeclareConstraint(rule_type=row['TypeOfConstraint'], activation=row['B'], target=row['A'])
                if dc not in declare_constraints:
                    declare_constraints.append(dc)

            return declare_constraints

        except Exception as e:
            Print.RED.print(e)
            cnxn.close()
            return None