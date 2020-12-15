#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from allennlp.common import Registrable

from utils.wsd import WSDInstance, to_bn_id


class WSDInstanceConversionStrategy(Registrable):

    def convert(self, wsd_instance: WSDInstance) -> WSDInstance:
        raise NotImplementedError


@WSDInstanceConversionStrategy.register('identity')
class IdentityWSDInstanceConversionStrategy(WSDInstanceConversionStrategy):

    def convert(self, wsd_instance: WSDInstance) -> WSDInstance:
        return wsd_instance


@WSDInstanceConversionStrategy.register('babelnet')
class BabelNetWSDInstanceConversionStrategy(WSDInstanceConversionStrategy):

    def convert(self, wsd_instance: WSDInstance) -> WSDInstance:
        return WSDInstance(
            annotated_token=wsd_instance.annotated_token,
            instance_id=wsd_instance.instance_id,
            labels=[to_bn_id(label) for label in wsd_instance.labels] if wsd_instance.labels is not None else None
        )

