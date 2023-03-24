"""Console script for pythoncv."""
import sys
import typer

app = typer.Typer()


@app.command()
def main(
    name: str = typer.Argument(..., help="Your name"),
    age: int = typer.Option(18, help="Your age"),
    is_cool: bool = typer.Option(False, help="Are you cool?"),
):
    """Console script for pythoncv."""
    typer.echo(f"Hello {name}, you are {age} years old and cool: {is_cool}")


if __name__ == "__main__":
    app()
