#!/usr/bin/env python3

import http.server
import socketserver
import ssl
import socket

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

def get_local_ip():
    try:
        # Connect to a remote server to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except:
        return "localhost"

PORT = 8443
Handler = MyHTTPRequestHandler

local_ip = get_local_ip()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # Create SSL context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    
    # Wrap the socket with SSL
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"HTTPS Server running on:")
    print(f"  Local: https://localhost:{PORT}")
    print(f"  Network: https://{local_ip}:{PORT}")
    print(f"\nTo access from other devices:")
    print(f"1. Open https://{local_ip}:{PORT} on your device")
    print(f"2. Accept the security warning (self-signed certificate)")
    print(f"3. Grant camera permissions when prompted")
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
