import sys
import time

import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import proto.mi_mensaje_pb2 as mi_mensaje_pb2

ecal_core.initialize(sys.argv, "Python Protobuf Publisher")

pub = ProtoPublisher("mi_mensaje", mi_mensaje_pb2.mi_mensaje)
protobuf_message = mi_mensaje_pb2.mi_mensaje()
counter = 0

while ecal_core.ok():
    message = input("Enter your comment: ")
    protobuf_message.comment = message
    protobuf_message.id = 123456
    protobuf_message.date = 321654987
    pub.send(protobuf_message)
    time.sleep(1)
    counter = counter + 1

ecal_core.finalize()
