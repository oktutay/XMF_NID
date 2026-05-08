"""
NFStream-based flow + packet feature extractor used by XMF-GNN.

Identical control-flow to the XG-NID GNN4ID extractor: an NFStreamer is
instantiated with ``udps=My_Custom(limit=20)`` so that every flow stops at
20 packets (paper Sec. 3.1: "we set a limit on the maximum number of
packets in a flow to 20"). Each packet contributes:

    payload_data       -- hex string of the payload bytes (1500-dim cap
                          applied at graph build time, paper Sec. 3.1)
    delta_time         -- ms since the previous packet of the flow
    packet_direction   -- 0 (src->dst) or 1 (dst->src)
    ip_size            -- IP packet size in bytes
    transport_size     -- transport-layer size in bytes
    payload_size       -- payload size in bytes
    syn / cwr / ece / urg / ack / psh / rst / fin
                       -- TCP flag bits

These together form the 14 packet-level protocol features mentioned in
paper Sec. 3.1 + Sec. 3.2; the 1500-dim payload byte vector is the
content-side feature.

Run as a CLI to convert a single PCAP file into a flow CSV:

    python Feature_extractor_flow_packet_combined.py path/to/in.pcap path/to/out_dir/

The output CSV is consumed by ``Additional_Features.additional_features()``
to add the 52-dim multi-scale temporal block, then by ``NIDSDataset`` to
build heterogeneous graph objects.
"""

from __future__ import annotations

import argparse
import os

from nfstream import NFPlugin, NFStreamer


class My_Custom(NFPlugin):

    def on_init(self, packet, flow):
        if self.limit == 1:
            flow.expiration_id = -1

        flow.udps.payload_data = []
        if packet.payload_size > 0:
            flow.udps.payload_data.append(
                packet.ip_packet[-packet.payload_size:].hex()
            )
        else:
            flow.udps.payload_data.append("00")

        flow.udps.delta_time = [packet.delta_time]
        flow.udps.packet_direction = [packet.direction]
        flow.udps.ip_size = [packet.ip_size]
        flow.udps.transport_size = [packet.transport_size]
        flow.udps.payload_size = [packet.payload_size]

        flow.udps.syn = [packet.syn]
        flow.udps.cwr = [packet.cwr]
        flow.udps.ece = [packet.ece]
        flow.udps.urg = [packet.urg]
        flow.udps.ack = [packet.ack]
        flow.udps.psh = [packet.psh]
        flow.udps.rst = [packet.rst]
        flow.udps.fin = [packet.fin]

    def on_update(self, packet, flow):
        if packet.payload_size > 0:
            flow.udps.payload_data.append(
                packet.ip_packet[-packet.payload_size:].hex()
            )
        else:
            flow.udps.payload_data.append("00")

        flow.udps.delta_time.append(packet.delta_time)
        flow.udps.packet_direction.append(packet.direction)
        flow.udps.ip_size.append(packet.ip_size)
        flow.udps.transport_size.append(packet.transport_size)
        flow.udps.payload_size.append(packet.payload_size)

        flow.udps.syn.append(packet.syn)
        flow.udps.cwr.append(packet.cwr)
        flow.udps.ece.append(packet.ece)
        flow.udps.urg.append(packet.urg)
        flow.udps.ack.append(packet.ack)
        flow.udps.psh.append(packet.psh)
        flow.udps.rst.append(packet.rst)
        flow.udps.fin.append(packet.fin)

        if self.limit == flow.bidirectional_packets:
            flow.expiration_id = -1  # force expiration at limit packets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run NFStream over a PCAP file and emit a CSV with both flow "
            "and per-packet (udps.*) columns, ready for "
            "Additional_Features.additional_features()."
        )
    )
    parser.add_argument("pcap_files", help="Path to a PCAP file.")
    parser.add_argument("Destination_path",
                        help="Directory where the extracted CSV is stored.")
    args = parser.parse_args()

    streamer = NFStreamer(
        source=args.pcap_files,
        accounting_mode=1,
        idle_timeout=120,                 # paper Sec. 3.1: idle 120 s
        statistical_analysis=True,
        n_dissections=0,
        udps=My_Custom(limit=20),         # paper Sec. 3.1: max 20 packets / flow
    )
    print("*** Done Reading ***")

    name = os.path.basename(args.pcap_files).split(".")[0]
    out = os.path.join(args.Destination_path, name + ".csv")
    streamer.to_csv(path=out, columns_to_anonymize=[], flows_per_file=0)
