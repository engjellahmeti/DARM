# @project Deviance Analysis by Means of Redescription Mining - Master Thesis

# @author Volodymyr Leno, Marlon Dumas, Fabrizio Maria Maggi, Marcello La Rosa,Artem Polyvyanyy
# @author Engjëll Ahmeti, updated as per need of the project
# @date 12/8/2020

from feature_vectors.feature_vector import FeatureVector
from event_log_reader.reader import EventLogReader
import pandas as pd
from log_print import Print
from typing import Tuple, List
from pandas import DataFrame
from redescription_mining.data_model import RedescriptionDataModel

class RuleExtractor:
    def extract_fulfilment(self, event_log_path: str, declare_constraint: str, is_positive_or_negative_log: str,
                           remove_attributes: bool = False, write_to_CSV: bool = True) -> RedescriptionDataModel:
        event_log_reader = EventLogReader()

        (_, cases) = event_log_reader.gain_log_info_table(event_log_path)

        feature_vectors: List[FeatureVector] = []

        Print.YELLOW.print(
            "Started creating 'Constraint Instances' and after that 'Feature Vectors' for {0} event log. ".format(
                is_positive_or_negative_log))
        for caseID in cases.keys():
            id_a = []
            id_t = []

            k = 0
            for event in cases[caseID]:
                if event.activity_name in declare_constraint.activation:
                    id_a.append(k)
                if event.activity_name in declare_constraint.target:
                    id_t.append(k)

                k += 1

            if declare_constraint.rule_type == "response" or declare_constraint.rule_type == "chainresponse":
                for i in id_a:
                    for j in id_t:
                        if ((declare_constraint.rule_type == "response" and j > i) or (
                                declare_constraint.rule_type == "chainresponse" and (i - j == -1))):
                            feature_vectors.append(
                                FeatureVector(fromm=cases[caseID][i].payload, to=cases[caseID][j].payload, remove_some_attributes=remove_attributes,
                                              trace=caseID))
                            break

            elif declare_constraint.rule_type == "chainprecedence":
                for i in id_a:
                    for j in id_t:
                        if declare_constraint.rule_type == "chainprecedence" and (j - i == -1):
                            feature_vectors.append(
                                FeatureVector(fromm=cases[caseID][i].payload, to=cases[caseID][j].payload, remove_some_attributes=remove_attributes,
                                              trace=caseID))
                            break

            elif declare_constraint.rule_type == "precedence":
                temp = None
                for i in id_a:
                    for j in id_t:
                        if i > j:
                            temp = (i, j)
                        if j >= i:
                            break
                    if temp:
                        feature_vectors.append(
                            FeatureVector(fromm=cases[caseID][temp[0]].payload, to=cases[caseID][temp[1]].payload,
                                          remove_some_attributes=remove_attributes, trace=caseID))
                    temp = None

            elif declare_constraint.rule_type == "respondedexistence":
                for i in id_a:
                    for j in id_t:
                        if j != i:
                            feature_vectors.append(
                                FeatureVector(fromm=cases[caseID][i], to=cases[caseID][j], remove_some_attributes=remove_attributes, trace=caseID))

            elif declare_constraint.rule_type == "alternateresponse":
                for i in id_a:
                    for j in id_t:
                        none_match = [(el > i and el < j) for el in id_t]
                        if j > i and not (True in none_match):
                            feature_vectors.append(
                                FeatureVector(fromm=cases[caseID][i], to=cases[caseID][j], remove_some_attributes=remove_attributes, trace=caseID))

            elif declare_constraint.rule_type == "alternateprecedence":
                for i in id_a:
                    for j in id_t:
                        none_match = [(el < i and el > j) for el in id_t]

                        if j < i and not (True in none_match):
                            feature_vectors.append(
                                FeatureVector(fromm=cases[caseID][i].payload, to=cases[caseID][j].payload, remove_some_attributes=remove_attributes,
                                              trace=caseID))

        if write_to_CSV:
            redescription_data_model: RedescriptionDataModel = self.write_to_CSV_(feature_vectors=feature_vectors,
                                                                            is_positive_or_negative_log=is_positive_or_negative_log)

            Print.YELLOW.print("'Feature Vectors' have been created.")

            return redescription_data_model


        Print.YELLOW.print("'Feature Vectors' have been created.")
        return feature_vectors, None, None, None

    def convert_feature_vectors_into_activation_and_target_frames(self, feature_vectors: List[FeatureVector]) -> Tuple[DataFrame, DataFrame]:
        if len(feature_vectors) > 0:
            activation_cols = feature_vectors[0].fromm.keys()
            activation_frame = pd.DataFrame(columns=activation_cols)
            target_cols = feature_vectors[0].to.keys()
            target_frame = pd.DataFrame(columns=target_cols)

        index = 0
        for row in feature_vectors:
            activation_app_yellow = []
            for key in activation_cols:
                activation_app_yellow = activation_app_yellow + [row.fromm[key]]
            activation_frame.loc[index] = activation_app_yellow

            target_app_yellow = []
            for key in target_cols:
                target_app_yellow = target_app_yellow + [row.to[key]]
            target_frame.loc[index] = target_app_yellow

            index += 1

        return (activation_frame, target_frame)

    def convert_feature_vectors_into_activation_and_target_frames_for_traces(self, feature_vectors: List[FeatureVector]) -> Tuple[DataFrame, DataFrame]:
        if len(feature_vectors) > 0:
            activation_cols = ['Trace'] + list(feature_vectors[0].fromm.keys())
            activation_frame = pd.DataFrame(columns=activation_cols)
            target_cols = ['Trace'] + list(feature_vectors[0].to.keys())
            target_frame = pd.DataFrame(columns=target_cols)

        index = 0
        for row in feature_vectors:
            activation_app_yellow = [row.trace]
            for key in activation_cols:
                if key != 'Trace':
                    activation_app_yellow = activation_app_yellow + [row.fromm[key]]
            activation_frame.loc[index] = activation_app_yellow

            target_app_yellow = [row.trace]
            for key in target_cols:
                if key != 'Trace':
                    target_app_yellow = target_app_yellow + [row.to[key]]
            target_frame.loc[index] = target_app_yellow

            index += 1

        return (activation_frame, target_frame)

    def write_to_CSV_(self, feature_vectors: List[FeatureVector], is_positive_or_negative_log: str) -> RedescriptionDataModel:
        activation_path = 'feature_vectors/csv_feature_vectors/' + is_positive_or_negative_log + '/activation.csv'
        target_path = 'feature_vectors/csv_feature_vectors/' + is_positive_or_negative_log + '/target.csv'

        if len(feature_vectors) > 0:
            (activation_frame, target_frame) = self.convert_feature_vectors_into_activation_and_target_frames(feature_vectors=feature_vectors)

          

            activation_frame.to_csv(activation_path)
            target_frame.to_csv(target_path)

            if is_positive_or_negative_log == 'negative':
                (activation_frameTrace, target_frameTrace) = self.convert_feature_vectors_into_activation_and_target_frames_for_traces(
                    feature_vectors=feature_vectors)
                merged_trace = 'feature_vectors/csv_feature_vectors/' + is_positive_or_negative_log + '/traces.csv'
                merged = pd.merge(how='inner', on='Trace', left=activation_frameTrace, right=target_frameTrace)
                merged.drop_duplicates(subset=['Trace'], inplace=True)
                merged.to_csv(merged_trace)

            return RedescriptionDataModel(activation_view=activation_path, activation_attributes=list(activation_frame.columns), target_view=target_path, target_attributes=list(target_frame.columns))

        else:
            with open(activation_path, 'wt') as a:
                a.write('')
            with open(target_path, 'wt') as a:
                a.write('')

            return RedescriptionDataModel(activation_view=activation_path, activation_attributes=[], target_view=target_path, target_attributes=[])
            