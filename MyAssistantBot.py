import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext
from telegram.ext.filters import TEXT
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with your BotFather-provided API Token
API_TOKEN = '8153326149:AAH52pyyaYqapzXNLJoBHFSkyYtd6GzugB0'

# Load the TinyLlama model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set up the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate a response from the model
def generate_response(prompt):
    formatted_prompt = f"Question: {prompt}\nAnswer:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response}")
    return response


# Handle user messages
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    print(f"Received message: {user_message}")

    # Generate response from the model
    response = generate_response(user_message)
    print(f"Generated response: {response}")

    # Send the response back to the user
    await update.message.reply_text(response)

# Handle /start command
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hello! I am your AI Assistant. Send me a message!")

# Initialize the Bot
def main():
    # Build the application
    app = Application.builder().token(API_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))

    # Run using a standalone event loop
    print("Bot is running... Press Ctrl+C to stop.")
    try:
        asyncio.run(app.run_polling())
    except RuntimeError:
        # If there is an existing event loop in the runtime environment, use this method
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app.run_polling())

if __name__ == "__main__":
    main()

