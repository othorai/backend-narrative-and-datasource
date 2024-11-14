#config.py
from pydantic_settings import BaseSettings
from typing import List
from urllib.parse import quote_plus

class Settings(BaseSettings):
    # Database settings
    DB_PASSWORD: str
    DB_NAME: str
    DB_HOST: str
    DB_PORT: str
    DB_USER: str

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # Other settings
    DEBUG: bool
    ALLOWED_HOSTS: str

    # OpenAI settings
    OPENAI_API_KEY: str  # Add this line

    @property
    def DATABASE_URL(self):
        url = f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"
        print(f"Constructed DATABASE_URL: {url}")  # Debug print
        return url

    @property
    def ALLOWED_HOSTS_LIST(self) -> List[str]:
        return [host.strip() for host in self.ALLOWED_HOSTS.split(',')]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

MYSQL_HOST = 'your_mysql_host'
MYSQL_USER = 'your_mysql_user'
MYSQL_PASSWORD = 'your_mysql_password'
MYSQL_DATABASE = 'your_mysql_database'

PG_HOST = 'your_postgresql_host'
PG_USER = 'your_postgresql_user'
PG_PASSWORD = 'your_postgresql_password'
PG_DATABASE = 'your_postgresql_database'

GOOGLE_SHEETS_CREDENTIALS_FILE = 'path/to/your/credentials.json'
GOOGLE_SHEETS_SPREADSHEET_ID = 'your_spreadsheet_id'

SALESFORCE_USERNAME = 'your_salesforce_username'
SALESFORCE_PASSWORD = 'your_salesforce_password'
SALESFORCE_SECURITY_TOKEN = 'your_salesforce_security_token'