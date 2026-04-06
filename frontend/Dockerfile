FROM nginx:alpine

RUN mkdir -p /etc/nginx/ssl

ARG FHIR_URL=http://localhost:8080/fhir
RUN printf '%s\n' \
    'window.AI_API_PREFIX="/ml-api";' \
    "window.FHIR_URL=\"${FHIR_URL}\";" \
    > /usr/share/nginx/html/config.js

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html app.js /usr/share/nginx/html/

EXPOSE 3000 3443
