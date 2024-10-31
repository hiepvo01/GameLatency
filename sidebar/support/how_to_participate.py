import streamlit as st
from datetime import datetime
from PIL import Image
import qrcode
import io

def show_how_to_participate():
    # st.title("Participate in Our Research")
    st.write("We're conducting exciting experiments to further our understanding of gaming performance.")

    tab1, tab2 = st.tabs(["UTS Campus Experiments", "Large Event Experiments"])

    with tab1:
        show_uts_experiments()

    with tab2:
        show_event_experiments()

def show_event_experiments():
    # Main title
    st.header("Large Event Experiments")

    # Add poster image
    poster_image = Image.open('assets/largeevent.png')
    st.image(poster_image, caption='Event Poster', use_column_width=True)

    # Upcoming Events
    st.subheader("Upcoming Events")
    st.write("""
    Get ready for an exciting series of large-scale gaming experiments! As part of our ongoing research on the impact of latency in first-person shooting (FPS) games, we're hosting several large event experiments that will allow you to experience gaming in a controlled environment, contribute to cutting-edge research, and connect with other gamers.
    """)

    # Event Details
    st.subheader("Event Details")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Event Information**")
        st.write("**Event Name:** [Insert Event Name]")
        st.write("**Date:** TBC")
        st.write("**Time:** TBC")
        st.write("**Location:** University of Technology Sydney (UTS) - TBC")

    with col2:
        st.write("**What to Expect**")
        st.markdown("""
        * **Structured Gameplay:** Participate in multiple gaming sessions where you'll experience different latency conditions and see how they affect your performance.
        * **Data Collection:** Your gameplay data will be recorded and analyzed as part of our research. This is your chance to contribute valuable insights to the future of online gaming.
        * **Networking Opportunities:** Meet fellow gamers, researchers, and industry professionals. Share your experiences and learn from others in the gaming community.
        * **Prizes and Giveaways:** As a token of our appreciation, participants will have the chance to win gaming-related prizes and receive exclusive giveaways.
        """)

    # How to Participate
    st.subheader("How to Participate")
    st.write("Interested in joining our large event experiments? Here's how you can get involved:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        * **Pre-Register:** Ensure your spot by pre-registering online. Space is limited, so we recommend signing up early.
        * **Walk-In Registration:** Limited walk-in spots will be available on the day of the event. Arrive early to secure your place.
        * **Group Participation:** Bring your friends! Group registrations are welcome. Indicate your group size when you pre-register.
        """)

    with col2:
        if st.button("Pre-Register Now"):
            st.success("Thank you for your interest! Pre-registration will be available soon.")

    # Who Can Participate
    st.subheader("Who Can Participate?")
    st.markdown("""
    * **Eligibility:** Open to all gamers aged 18 and above, regardless of experience level. Whether you're a seasoned pro or new to FPS games, your participation is valuable.
    * **Requirements:** Participants should be able to attend the UTS campus on the event date. No special gaming equipment is needed; everything will be provided.
    """)

    # Why Participate
    st.subheader("Why Participate?")
    st.markdown("""
    * **Contribute to Research:** Help us understand the effects of network latency on gameplay. Your participation will provide crucial data that could shape the future of gaming.
    * **Improve Your Skills:** Experience firsthand how latency affects your gameplay and learn strategies to improve your performance.
    * **Enjoy the Community:** Take part in a fun and engaging event, meet new people, and enjoy a day dedicated to gaming and research.
    """)

    # Questions
    st.subheader("Questions?")
    st.write("If you have any questions about the upcoming events, please don't hesitate to reach out to us at email@example.com. We're happy to provide more information and assist with your registration.")

    # Footer
    st.markdown("---")
    st.write("© 2024 Large Event Experiments - Gaming Latency Research")

def show_uts_experiments():
    # Main title
    st.header("UTS Campus Experiments")
    # Add poster image
    poster_image = Image.open('assets/utsevent.png')
    st.image(poster_image, caption='Event Poster', use_column_width=True)

    # Subtitle
    st.subheader("Welcome to the Gaming Latency Impact Study")

    # About the Study
    st.write("**About the Study**")
    st.write("""
    Are you a gamer interested in how network latency affects your performance? We invite you to participate in an exciting research study conducted by the University of Technology Sydney (UTS). This study explores how different levels of latency impact gameplay in first-person shooting (FPS) games, with the aim of improving gaming experiences for everyone.
    """)

    # Why Participate
    st.write("**Why Participate?**")
    st.markdown("""
    * **Contribute to Cutting-Edge Research:** Your participation will help us better understand the effects of latency on gaming performance and player experience, contributing to the future of online gaming.
    * **Enhance Your Gaming Skills:** Through structured gameplay sessions, you'll gain insights into how latency affects your performance, potentially improving your strategies in FPS games.
    * **Receive Compensation:** As a thank you for your time, participants will receive a $30 gift card or voucher.
    """)

    # What to Expect
    st.write("**What to Expect**")
    st.markdown("""
    * **Duration:** Each session will last approximately 3 hours, including gameplay, breaks, and surveys.
    * **Location:** All sessions will be held at the UTS campus in a dedicated gaming lab.
    * **Activities:** You will play multiple rounds of a popular FPS game under different latency conditions, followed by brief surveys to capture your experience.
    """)

    # Who Can Participate
    st.write("**Who Can Participate?**")
    st.markdown("""
    * **Eligibility:** We are looking for both experienced and novice FPS gamers aged 18 and above. Whether you're a seasoned player or just getting started, your input is valuable.
    * **Requirements:** You should be available to visit the UTS campus for the experimental session. No special equipment or prior experience with the specific game is required.
    """)

    # How to Register
    st.subheader("How to Register")
    st.write("To express your interest in participating, you have two easy options:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        1. **Fill Out the Registration Form:** Click the button below to complete the online registration form. Once we receive your registration, we will contact you with more details and available time slots.
        """)
        if st.button("Register Now"):
            st.markdown("[Registration Form](https://forms.gle/4ycjRbZMy9zcZwiXA)")

    with col2:
        st.markdown("""
        2. **Scan the QR Code:** If you have the poster with you, simply scan the QR code with your smartphone. It will take you directly to the registration page where you can sign up in just a few clicks.
        """)
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data('https://forms.gle/4ycjRbZMy9zcZwiXA')
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        byte_im = buf.getvalue()

        # Display QR code
        st.image(byte_im, caption='Scan to Register', width=200)

    # Questions
    st.subheader("Questions?")
    st.write("If you have any questions or need more information, feel free to reach out to our research team at email@example.com.")
