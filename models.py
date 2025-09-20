
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class ProcessedDocument(Base):
    __tablename__ = "processed_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    image_hash = Column(String, unique=True, index=True)
    prompt = Column(Text)
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    file_name = Column(String)

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key_name = Column(String)
    month_year = Column(String)  # Format: YYYY-MM
    request_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
