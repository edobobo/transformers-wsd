#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import os
import xml.etree.cElementTree as ET
from typing import List, Optional, Iterable, Callable, Tuple, NamedTuple, Set

from nltk.corpus import wordnet as wn
from xml.dom import minidom


def wn_sense_key_to_id(sense_key):
    synset = wn.lemma_from_key(sense_key).synset()
    return 'wn:' + str(synset.offset()).zfill(8) + synset.pos()


_wn2bn = {}
_bn2wn = {}


with open('data/kb-mappings/bn2wn.txt') as f:
    for line in f:
        line = line.strip()
        parts = line.split('\t')
        _bn2wn[parts[0]] = parts[2]
        _wn2bn[parts[2]] = parts[0]


def wn_id2bn_id(wn_id):
    return _wn2bn[wn_id]


def bn_id2wn_id(bn_id):
    return _bn2wn[bn_id]


_to_bn_id_cache = {}


def to_bn_id(key):

    if key.startswith('bn:'):
        key_type = 'bn_id'
        transform = lambda x: x
    elif key.startswith('wn:'):
        key_type = 'wn_id'
        transform = lambda x: wn_id2bn_id(x)
    else:
        key_type = 'sense_key'
        transform = lambda x: to_bn_id(wn_sense_key_to_id(x).replace('s', 'a'))

    if key_type not in _to_bn_id_cache:
        _to_bn_id_cache[key_type] = {}

    if key not in _to_bn_id_cache[key_type]:
        _to_bn_id_cache[key_type][key] = transform(key)

    return _to_bn_id_cache[key_type][key]


class AnnotatedToken(NamedTuple):
    text: str
    pos: Optional[str] = None
    lemma: Optional[str] = None


class WSDInstance(NamedTuple):
    annotated_token: AnnotatedToken
    instance_id: Optional[str]
    labels: Optional[List[str]]


def read_from_raganato(
        xml_path: str,
        key_path: Optional[str] = None,
        instance_transform: Optional[Callable[[WSDInstance], WSDInstance]] = None
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:

    def read_by_text_iter(xml_path: str):

        it = ET.iterparse(xml_path, events=('start', 'end'))
        _, root = next(it)

        for event, elem in it:
            if event == 'end' and elem.tag == 'text':
                document_id = elem.attrib['id']
                for sentence in elem:
                    sentence_id = sentence.attrib['id']
                    for word in sentence:
                        yield document_id, sentence_id, word

            root.clear()

    mapping = {}

    if key_path is not None:
        with open(key_path) as f:
            for line in f:
                line = line.strip()
                wsd_instance, *labels = line.split(' ')
                mapping[wsd_instance] = labels

    last_seen_document_id = None
    last_seen_sentence_id = None

    for document_id, sentence_id, element in read_by_text_iter(xml_path):

        if last_seen_sentence_id != sentence_id:

            if last_seen_sentence_id is not None:
                yield last_seen_document_id, last_seen_sentence_id, sentence

            sentence = []
            last_seen_document_id = document_id
            last_seen_sentence_id = sentence_id

        annotated_token = AnnotatedToken(
            text=element.text,
            pos=element.attrib.get('pos', None),
            lemma=element.attrib.get('lemma', None)
        )

        wsd_instance = WSDInstance(
            annotated_token=annotated_token,
            instance_id=None if element.tag == 'wf' or element.attrib['id'] not in mapping else element.attrib['id'],
            labels=None if element.tag == 'wf' or element.attrib['id'] not in mapping else mapping[element.attrib['id']]
        )

        if instance_transform is not None:
            wsd_instance = instance_transform(wsd_instance)

        sentence.append(wsd_instance)

    yield last_seen_document_id, last_seen_sentence_id, sentence


class RaganatoBuilder:

    def __init__(self, lang: Optional[str] = None, source: Optional[str] = None):
        self.corpus = ET.Element('corpus')
        self.current_text_section = None
        self.current_sentence_section = None
        self.gold_senses = []

        if lang is not None:
            self.corpus.set('lang', lang)

        if source is not None:
            self.corpus.set('source', source)

    def open_text_section(self, text_id: str, text_source: str = None):
        text_section = ET.SubElement(self.corpus, 'text')
        text_section.set('id', text_id)
        if text_source is not None:
            text_section.set('source', text_source)
        self.current_text_section = text_section

    def open_sentence_section(self, sentence_id: str):
        sentence_section = ET.SubElement(self.current_text_section, 'sentence')
        sentence_id = self.compute_id([self.current_text_section.attrib['id'], sentence_id])
        sentence_section.set('id', sentence_id)
        self.current_sentence_section = sentence_section

    def add_annotated_token(self, token: str, lemma: str, pos: str, instance_id: Optional[str] = None, labels: Optional[List[str]] = None):
        if instance_id is not None and labels is not None:
            token_element = ET.SubElement(self.current_sentence_section, 'instance')
            token_id = self.compute_id([self.current_sentence_section.attrib['id'], instance_id])
            token_element.set('id', token_id)
            self.gold_senses.append((token_id, ' '.join(labels)))
        else:
            token_element = ET.SubElement(self.current_sentence_section, 'wf')
        token_element.set('lemma', lemma)
        token_element.set('pos', pos)
        token_element.text = token

    @staticmethod
    def compute_id(chain_ids: List[str]) -> str:
        return '.'.join(chain_ids)

    def store(self, data_output_path: str, labels_output_path: str, prettify: bool = True):
        self.__store_xml(data_output_path, prettify)
        self.__store_labels(labels_output_path)

    def __store_xml(self, output_path: str, prettify: bool):
        corpus_writer = ET.ElementTree(self.corpus)
        with open(output_path, 'wb') as f_xml:
            corpus_writer.write(f_xml, encoding='UTF-8', xml_declaration=True)
        if prettify:
            dom = minidom.parse(output_path)
            pretty_xml = dom.toprettyxml()
            with open(output_path, 'w') as f_xml:
                f_xml.write(pretty_xml)

    def __store_labels(self, output_path: str):
        with open(output_path, 'w') as f_labels:
            for gold_sense in self.gold_senses:
                f_labels.write(' '.join(gold_sense))
                f_labels.write('\n')


def expand_raganato_path(path: str) -> Tuple[str, str]:
    return f'{path}.data.xml', f'{path}.gold.key.txt'

