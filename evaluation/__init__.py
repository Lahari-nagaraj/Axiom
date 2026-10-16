"""Axiom evaluation harness.

This package is intentionally isolated from the production app. It imports the
existing application modules (config, vectorstore, self_rag) without modifying
them, and never touches the chat UI.
"""
